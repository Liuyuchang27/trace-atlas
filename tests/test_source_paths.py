import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import SystemMessage
from test_core_evidence import setup

from news_agent.evidence_chain import claim_needed, unstarted_main_goals
from news_agent.harness import ResearchBoundary, make_tools
from news_agent.progress import plan_tool
from news_agent.retrieval import LocalCorpus
from news_agent.schemas import DocumentVersion
from news_agent.source_paths import discover_links, pending_routes, source_link_tool


def test_link_routes_are_exact_original_targets_not_automatically_read_evidence(tmp_path, task):
    e, b, context = setup(tmp_path, task)
    doc = DocumentVersion(
        version_id="old-case",
        title="Aurora case record",
        url="https://regulator.example/old-case",
        content=(
            "[Aurora contents](#contents)\n\n"
            "A separate Aurora investigation: [New Aurora case](/new-case).\n\n"
            "[Aurora unsafe](http://127.0.0.1/private)\n\n"
            "[Aurora instructions](javascript:alert)"
        ),
    )
    e.add_document(doc)
    selected = discover_links(e, b, doc.version_id, "Aurora separate investigation")
    assert len(selected) == 1
    row = selected[0]
    assert row["url"] == "https://regulator.example/new-case"
    assert doc.content[row["start"] : row["end"]] == row["original_link"]
    assert "passage_id" not in row
    packet = context.packet([])
    assert packet["source_link_leads"][0]["url"] == row["url"]
    assert not packet["documents"][0]["excerpts"]
    assert len(e.documents) == 1 and b.state.get("retrieval_api_calls", 0) == 0
    b.finish_research("test")
    assert not context.packet([])["source_link_leads"]
    assert "research_finished" in asyncio.run(
        source_link_tool(e, b).ainvoke({"version_id": doc.version_id, "query": "Aurora"})
    )


def test_earlier_unread_candidates_survive_a_new_search_without_becoming_evidence(tmp_path, task):
    e, b, _ = setup(tmp_path, task)
    hits = [
        {"version_id": str(i), "title": "Candidate " + str(i), "url": f"https://s{i}.example/"}
        for i in range(10)
    ]
    b.state["search_candidates"] = {h["version_id"]: h for h in hits}
    b.state["last_search_results"] = hits[5:]
    candidates, _ = pending_routes(e, b)
    assert len(candidates) == 5
    assert any(int(h["version_id"]) < 5 for h in candidates)
    assert any(int(h["version_id"]) >= 5 for h in candidates)
    assert not e.documents and not b.state.get("delivered_passages")


def test_first_read_delivers_date_heading_with_the_original_event(tmp_path, task):
    e, b, _ = setup(tmp_path, task)
    task = task.model_copy(update={"mode": "retrospective"})
    doc = DocumentVersion(
        version_id="dated",
        title="Aurora project closing record",
        url="https://issuer.example/closing",
        content=(
            "# 30 October 2026\n\nAurora has completed the project closing.\n\nUnrelated prose."
        ),
    )
    tools = {t.name: t for t in make_tools(LocalCorpus([doc], task), e, b)}
    response = json.loads(asyncio.run(tools["read_document"].ainvoke({"version_id": "dated"})))
    assert any(p["text"] == "# 30 October 2026" for p in response["passages"])
    assert all(doc.content[p["start"] : p["end"]] == p["text"] for p in response["passages"])


def test_claim_fallback_does_not_interrupt_research_or_duplicate_existing_findings(tmp_path, task):
    e, b, _ = setup(tmp_path, task)
    doc = DocumentVersion(
        version_id="one", title="Record", url="https://issuer.example/one", content="Original."
    )
    p = e.add_document(doc)[0]
    b.state["evidence_focus"] = {p.passage_id: {"score": 1}}
    assert not claim_needed(e, b)
    b.finish_research("test")
    assert claim_needed(e, b)
    b.state["core_claim_turns"] = 1
    assert not claim_needed(e, b)
    b.state["core_claim_turns"] = 0
    e.add_claim("Recorded finding", [p.passage_id], None, None)
    assert not claim_needed(e, b)


def test_boundary_includes_system_prompt_before_exposing_research_tools(tmp_path, task):
    e, b, _ = setup(tmp_path, task)
    tools = make_tools(LocalCorpus([], task), e, b)
    b.state["tokens"] = 45000
    request = SimpleNamespace(
        messages=[],
        system_message=SystemMessage("X" * 16000),
        tools=tools,
        tool_choice=None,
        model_settings={},
    )
    request.override = lambda **kwargs: SimpleNamespace(**{**vars(request), **kwargs})
    observed = []

    async def handle(adapted):
        observed.extend(t.name for t in adapted.tools)

    boundary = ResearchBoundary(b, e.store, {t.name for t in tools}, evidence=e)
    asyncio.run(boundary.awrap_model_call(request, handle))
    assert b.state["phase"] == "writing"
    assert "search" not in observed and "read_document" not in observed


def test_model_declared_main_goal_schedules_one_search_before_more_background(tmp_path, task):
    e, b, _ = setup(tmp_path, task)
    doc = DocumentVersion(
        version_id="background",
        title="Earlier record",
        url="https://regulator.example/old",
        content="The earlier consultation remained conditional.",
    )
    e.add_document(doc)
    publish = plan_tool(e, b)
    response = asyncio.run(
        publish.ainvoke(
            {
                "steps": [
                    {
                        "step_id": "date",
                        "title": "Read dated consultation",
                        "status": "in_progress",
                        "answer_role": "date_anchor",
                    },
                    {
                        "step_id": "main",
                        "title": "Find an issuer original for the main status",
                        "status": "pending",
                        "answer_role": "main_status",
                    },
                ]
            }
        )
    )
    assert "saved_steps" in response and unstarted_main_goals(e, b)[0]["step_id"] == "main"
    tools = make_tools(LocalCorpus([], task), e, b)
    request = SimpleNamespace(
        messages=[],
        system_message=SystemMessage("Rules"),
        tools=tools,
        tool_choice=None,
        model_settings={},
    )
    request.override = lambda **kw: SimpleNamespace(**{**vars(request), **kw})
    seen = []

    async def handler(adapted):
        seen.append(adapted)

    boundary = ResearchBoundary(b, e.store, {t.name for t in tools}, evidence=e)
    asyncio.run(boundary.awrap_model_call(request, handler))
    assert [t.name for t in seen[0].tools] == ["search"]
    assert seen[0].tool_choice["function"]["name"] == "search"
    assert "Find an issuer original" in seen[0].messages[-1].content
    assert not unstarted_main_goals(e, b)  # bounded once, does not loop or certify completion
    assert not e.claims
    asyncio.run(boundary.awrap_model_call(request, handler))
    assert "read_document" in [t.name for t in seen[1].tools]
    b.state["main_goal_search_turns"] = 0
    b.finish_research("test")
    assert not unstarted_main_goals(e, b)
