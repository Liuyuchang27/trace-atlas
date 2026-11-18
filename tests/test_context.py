import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from news_agent.budget import Budget, BudgetExceeded
from news_agent.context import EvidenceContext, byte_size
from news_agent.harness import ResearchBoundary, make_tools
from news_agent.mock_model import FixtureChatModel
from news_agent.retrieval import LocalCorpus
from news_agent.schemas import BudgetConfig
from news_agent.storage import EvidenceStore, RunStore


def setup_context(tmp_path, task, corpus_path):
    store = RunStore(tmp_path)
    evidence = EvidenceStore(store)
    budget = Budget(BudgetConfig(), store)
    corpus = LocalCorpus.load(corpus_path, task)
    context = EvidenceContext(FixtureChatModel(), task, evidence, budget)
    return evidence, budget, corpus, context


def test_sdk_compaction_retains_task_ids_verbatim_evidence_and_drops_whole_tool_batch(
    tmp_path, task, corpus_path
):
    evidence, budget, corpus, context = setup_context(tmp_path, task, corpus_path)
    for doc in corpus.documents.values():
        evidence.add_document(doc)
    original = {k: p.model_dump() for k, p in evidence.passages.items()}
    messages = [
        HumanMessage(content=json.dumps({"task": task.model_dump(mode="json")})),
        AIMessage(content="", tool_calls=[{"name": "read_document", "args": {}, "id": "big"}]),
        ToolMessage(content="original-tool-transcript " * 1800, tool_call_id="big"),
    ]
    update = asyncio.run(context.abefore_model({"messages": messages}, None))
    assert update is not None
    # SDK removal marker + summary human message + safe continuation marker.
    assert len(update["messages"]) == 3
    assert all(not isinstance(m, ToolMessage) for m in update["messages"])
    summary = update["messages"][1]
    packet = json.loads(summary.content.split("\n\n", 1)[1])
    assert packet["task"] == task.model_dump(mode="json")
    assert byte_size(packet) <= context.limit
    ids = {p for row in packet["documents"] for p in row["passage_ids"]}
    assert ids == evidence.passages.keys()
    for row in packet["documents"]:
        for pid, text in row["excerpts"].items():
            assert evidence.passages[pid].text.startswith(text)
    assert budget.state["model_calls"] == 0
    assert budget.state["context_compactions"] == 1
    assert {k: p.model_dump() for k, p in evidence.passages.items()} == original
    assert "FUTURE_COMPLETION_SENTINEL" not in summary.content
    # Reopening the store after compaction reloads the exact original, not its prefix.
    restored = EvidenceStore(RunStore(tmp_path))
    assert {k: p.model_dump() for k, p in restored.passages.items()} == original


def test_compaction_preserves_unread_candidates_and_fails_closed_for_oversized_task(
    tmp_path, task, corpus_path
):
    evidence, budget, corpus, context = setup_context(tmp_path, task, corpus_path)
    tools = {t.name: t for t in make_tools(corpus, evidence, budget)}
    asyncio.run(tools["search"].ainvoke({"query": task.event}))
    packet = context.packet([])
    assert {c["version_id"] for c in packet["unread_candidates"]} == corpus.documents.keys()
    assert not packet["documents"]
    context.task = task.model_copy(update={"question": "x" * 20000})
    with pytest.raises(BudgetExceeded, match="evidence_context_index_limit"):
        context.packet([])


def test_writing_turns_persist_and_eventually_force_final_answer(tmp_path, task, corpus_path):
    evidence, budget, corpus, _ = setup_context(tmp_path, task, corpus_path)
    tools = make_tools(corpus, evidence, budget)
    boundary = ResearchBoundary(budget, evidence.store, {t.name for t in tools})
    budget.finish_research("retrieval_no_new_versions")
    request = SimpleNamespace(
        messages=[HumanMessage("task")],
        tools=tools,
        model_settings={},
        override=lambda **kwargs: kwargs,
    )

    async def handler(request):
        return request

    initial = asyncio.run(boundary.awrap_model_call(request, handler))
    assert {t.name for t in initial["tools"]} == {"record_claim", "get_passage", "list_evidence"}
    for index in range(2):
        budget.before_model(str(index), 100)
        budget.after_model(str(index), 200)
    restored = Budget(BudgetConfig(), evidence.store)
    assert restored.state["writing_model_calls"] == 2
    boundary.budget = restored
    final = asyncio.run(boundary.awrap_model_call(request, handler))
    assert final["tools"] == []
    assert "all read evidence" in final["messages"][-1].content


def test_compaction_keeps_recently_reloaded_qualifier_after_the_lead_prefix(
    tmp_path, task, corpus_path
):
    evidence, budget, corpus, context = setup_context(tmp_path, task, corpus_path)
    doc = next(iter(corpus.documents.values())).model_copy(
        update={
            "content": "Background. " * 100
            + "Critical qualifier: this is only a plan, not completion."
        }
    )
    passage = evidence.add_document(doc)[0]
    assert "Critical qualifier" not in passage.text[:600]
    tools = {t.name: t for t in make_tools(corpus, evidence, budget)}
    result = json.loads(
        asyncio.run(tools["get_passage"].ainvoke({"passage_id": passage.passage_id}))
    )
    assert result["text"] == passage.text
    packet = context.packet([])
    assert packet["documents"][0]["excerpts"][passage.passage_id] == passage.text
    restored = Budget(BudgetConfig(), evidence.store)
    assert restored.state["recent_passages"][-1] == passage.passage_id


def test_repeat_page_returns_ids_and_specific_passage_can_be_reloaded(tmp_path, task, corpus_path):
    evidence, budget, corpus, _ = setup_context(tmp_path, task, corpus_path)
    tools = {t.name: t for t in make_tools(corpus, evidence, budget)}
    version = next(iter(corpus.documents))
    first = json.loads(asyncio.run(tools["read_document"].ainvoke({"version_id": version})))
    second = json.loads(asyncio.run(tools["read_document"].ainvoke({"version_id": version})))
    assert second["already_read"]
    pid = first["passages"][0]["passage_id"]
    assert pid in second["passage_ids"]
    original = json.loads(asyncio.run(tools["get_passage"].ainvoke({"passage_id": pid})))
    assert original == first["passages"][0]
    assert budget.state["tool_calls"] == 2


def test_compaction_retains_each_document_lead_when_recent_reads_favor_one_document(
    tmp_path, task, corpus_path
):
    evidence, budget, corpus, context = setup_context(tmp_path, task, corpus_path)
    template = next(iter(corpus.documents.values()))
    leads = {}
    for index in range(5):
        doc = template.model_copy(
            update={
                "version_id": f"coverage-{index}",
                "title": f"Document {index}",
                "content": (f"Original lead {index}. " + "x" * 1500)[:1500] + "y" * 1500,
            }
        )
        passages = evidence.add_document(doc)
        leads[passages[0].passage_id] = passages[0].text
        budget.state["recent_passages"] = [passages[-1].passage_id]
    packet = context.packet([])
    excerpts = {pid: text for row in packet["documents"] for pid, text in row["excerpts"].items()}
    assert all(excerpts.get(pid) == text for pid, text in leads.items())
    assert byte_size(packet) <= context.limit
