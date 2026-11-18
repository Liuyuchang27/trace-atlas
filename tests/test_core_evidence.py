import asyncio
import hashlib
import json

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from news_agent.budget import Budget, BudgetCallback, BudgetExceeded
from news_agent.context import byte_size
from news_agent.core_context import CoreEvidenceContext
from news_agent.evidence_chain import core_policy, expose_passages, rank_passages
from news_agent.harness import make_tools, remember_passages
from news_agent.mock_model import FixtureChatModel
from news_agent.progress import document_locator, visual_search_tool
from news_agent.retrieval import LocalCorpus
from news_agent.schemas import BudgetConfig, DocumentVersion
from news_agent.storage import EvidenceStore, RunStore, read_json


def setup(tmp_path, task):
    evidence = EvidenceStore(RunStore(tmp_path))
    budget = Budget(BudgetConfig(), evidence.store)
    budget.state["evidence_chain_version"] = core_policy()["version"]
    context = CoreEvidenceContext(FixtureChatModel(), task, evidence, budget)
    return evidence, budget, context


def document(vid="record"):
    return DocumentVersion(
        version_id=vid,
        title="Aurora project closing announcement",
        url="https://issuer.example/" + vid,
        available_at="2026-06-01T00:00:00+00:00",
        availability_proof="Synthetic fixture availability record",
        content=(
            "Accept all cookies. Privacy. Sign in. Home. "
            * 150
            + "\n\nAurora announced a proposed project, subject to approval. "
            "It is expected to close next year; this is not a completed closing."
            + "\n\n"
            + "Forward-looking statements about Aurora project plans and closing. " * 60
        ),
    )


def test_ranking_does_not_reward_repeated_footer_over_original_qualifier(tmp_path, task):
    e, _, _ = setup(tmp_path, task)
    originals = e.add_document(document())
    before = {p.passage_id: p.model_dump() for p in originals}
    ranked = rank_passages(originals, "Aurora proposed project approval expected close")
    p = ranked[0]["passage"]
    assert "not a completed closing" in p.text
    assert "Forward-looking" not in p.text and "cookies" not in p.text
    assert document().content[p.start : p.end] == p.text
    assert before == {p.passage_id: p.model_dump() for p in originals}


def test_first_read_delivers_body_and_claims_require_delivered_originals(tmp_path, task):
    e, b, _ = setup(tmp_path, task)
    doc = document()
    corpus = LocalCorpus([doc], task)
    tools = {t.name: t for t in make_tools(corpus, e, b)}
    result = json.loads(asyncio.run(tools["read_document"].ainvoke({"version_id": doc.version_id})))
    assert result["selection"]["method"] == "ranked_original_passages"
    assert any("not a completed closing" in p["text"] for p in result["passages"])
    unseen = next(pid for pid in e.passages if pid not in b.state["delivered_passages"])
    bad = asyncio.run(tools["record_claim"].ainvoke({"text": "Guess", "passage_ids": [unseen]}))
    assert "actually delivered" in bad and not e.claims
    body = next(p for p in result["passages"] if "not a completed closing" in p["text"])
    good = json.loads(
        asyncio.run(
            tools["record_claim"].ainvoke(
                {
                    "text": "The project is proposed and conditional, not documented as completed.",
                    "passage_ids": [body["passage_id"]],
                    "stage": "announcement",
                }
            )
        )
    )
    assert good["claim"]["verification"] == "unverified"
    assert e.documents[doc.version_id].content == doc.content


def test_core_packet_preserves_focused_originals_after_navigation_rereads(tmp_path, task):
    e, b, context = setup(tmp_path, task)
    expected = {}
    for index in range(3):
        doc = document(str(index))
        passages = e.add_document(doc)
        ranked = rank_passages(passages, "proposed project approval expected close")
        target = ranked[0]["passage"]
        expose_passages(
            e,
            b,
            [target],
            method="find_saved_text",
            query="proposed project approval expected close",
            scores={target.passage_id: ranked[0]},
        )
        expected[target.passage_id] = target.text
        remember_passages(b, [p.passage_id for p in passages[:4]])
    b.finish_research("research_budget_exhausted")
    packet = context.packet([])
    excerpts = {pid: text for d in packet["documents"] for pid, text in d["excerpts"].items()}
    assert all(excerpts.get(pid) == text for pid, text in expected.items())
    assert byte_size(packet) <= context.limit == 10000
    assert not packet["unread_candidates"]
    assert packet["inventory"]["saved_passages"] > packet["inventory"]["selected_original_passages"]
    assert "NOT absent" in packet["rules"]
    assert all(text == e.passages[pid].text for pid, text in excerpts.items())
    assert all(pid in b.state["delivered_passages"] for pid in excerpts)


def test_saved_but_unread_text_never_enters_core_packet(tmp_path, task):
    e, b, context = setup(tmp_path, task)
    e.add_document(document())
    packet = context.packet([])
    assert packet["inventory"]["saved_passages"] > 0
    assert packet["inventory"]["tool_delivered_passages"] == 0
    assert packet["documents"][0]["excerpts"] == {}
    assert not b.state.get("delivered_passages")


def test_exact_context_archive_is_bound_to_request_using_packet_id(tmp_path, task):
    e, b, context = setup(tmp_path, task)
    passages = e.add_document(document())
    p = rank_passages(passages, "proposed project approval")[0]["passage"]
    expose_passages(e, b, [p], method="ranked_read", query="proposed project approval")
    packet = context.packet([])
    context.archive_packet(packet, "agent")
    record = b.state["active_context_packet"]
    assert hashlib.sha256((tmp_path / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    assert read_json(tmp_path / record["path"]) == packet
    callback = BudgetCallback(b, 0)
    callback.on_chat_model_start(
        {}, [[HumanMessage(json.dumps(packet))]], run_id="bound", invocation_params={"tools": []}
    )
    b.after_model("bound", 10)
    callback.on_chat_model_start(
        {},
        [[HumanMessage("Separate auxiliary request")]],
        run_id="separate",
        invocation_params={"tools": []},
    )
    b.after_model("separate", 10)
    events = [json.loads(s) for s in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    uses = [x for x in events if x["kind"] == "model_context_used"]
    assert len(uses) == 1 and uses[0]["action_id"] == "bound"


def test_duplicate_locator_keeps_originals_without_redelivering_full_text(tmp_path, task):
    e, b, _ = setup(tmp_path, task)
    e.add_document(document())
    locate = document_locator(e, b)
    args = {"version_id": "record", "query": "expected close approval", "limit": 1}
    first = json.loads(asyncio.run(locate.ainvoke(args)))
    second = json.loads(asyncio.run(locate.ainvoke(args)))
    assert "passages" in first and "passages" not in second
    assert second["already_delivered"] is True
    assert first["passages"][0]["passage_id"] in b.state["evidence_focus"]
    assert b.state.get("retrieval_api_calls", 0) == 0


def test_bad_optional_plan_id_does_not_block_search_but_fake_evidence_does(tmp_path, task):
    e, b, _ = setup(tmp_path, task)
    calls = []

    @tool
    async def search(query: str, limit: int = 5) -> str:
        """Synthetic search response; no external request."""
        calls.append(query)
        return json.dumps({"results": []})

    wrapper = visual_search_tool(search, e, b)
    args = {
        "query": "status record",
        "purpose": "Find the missing status record",
        "intent": "gap",
        "step_id": "typo",
    }
    result = json.loads(asyncio.run(wrapper.ainvoke(args)))
    assert result["plan_link_warning"] and calls == ["status record"]
    bad = json.loads(asyncio.run(wrapper.ainvoke({**args, "based_on": ["p-fake"]})))
    assert "error" in bad and len(calls) == 1


def test_core_packet_refuses_oversized_task_instead_of_hiding_it(tmp_path, task):
    _, _, context = setup(tmp_path, task)
    context.task = task.model_copy(update={"question": "X" * 20000})
    with pytest.raises(BudgetExceeded, match="evidence_context_index_limit"):
        context.packet([])
