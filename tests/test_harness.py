import asyncio
import json

import pytest
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage

import news_agent.harness as harness
from news_agent.schemas import BudgetConfig
from news_agent.storage import read_json


def test_real_sdk_offline_run_and_idempotent_completed_resume(tmp_path, task, corpus_path):
    result = asyncio.run(
        harness.run_research(
            tmp_path,
            task,
            corpus_path,
            BudgetConfig(),
            mock=True,
        )
    )
    assert result["status"] == "complete", result
    assert result["engine"] == "deepagents"
    assert result["budget"]["model_calls"] > 1
    assert result["api_cost_usd"] == 0
    assert read_json(tmp_path / "timeline.json")
    report = (tmp_path / "report.md").read_text()
    assert "SCRIPTED MOCK" in report and "[p-" in report
    assert len(read_json(tmp_path / "evidence.json")["documents"]) == 3
    for p in tmp_path.iterdir():
        if p.is_file():
            data = p.read_bytes()
            assert b"FUTURE_COMPLETION_SENTINEL" not in data, p
            assert b"REVISED_CONTENT_SENTINEL" not in data, p
    before = (tmp_path / "trajectory.jsonl").read_bytes()
    again = asyncio.run(harness.run_research(tmp_path, resume=True))
    assert again == result
    assert (tmp_path / "trajectory.jsonl").read_bytes() == before


def test_resume_rejects_changed_corpus(tmp_path, task, corpus_path):
    asyncio.run(harness.run_research(tmp_path, task, corpus_path, BudgetConfig(), mock=True))
    (tmp_path / "input_corpus.json").write_text("[]")
    with pytest.raises(ValueError, match="corpus snapshot changed"):
        asyncio.run(harness.run_research(tmp_path, resume=True))


def test_interrupted_checkpoint_resumes_without_repeating_search(
    tmp_path, task, corpus_path, monkeypatch
):
    original = harness.create_deep_agent

    class InterruptOnce:
        def __init__(self, graph):
            self.graph = graph

        async def ainvoke(self, initial, config):
            return await self.graph.ainvoke(initial, config, interrupt_after=["tools"])

        async def aget_state(self, config):
            return await self.graph.aget_state(config)

    monkeypatch.setattr(harness, "create_deep_agent", lambda **kw: InterruptOnce(original(**kw)))
    first = asyncio.run(
        harness.run_research(
            tmp_path,
            task,
            corpus_path,
            BudgetConfig(),
            mock=True,
        )
    )
    assert first["status"] == "partial"
    assert first["stop_reason"] == "checkpoint_interrupted"
    deadline = first["budget"]["deadline_at"]
    monkeypatch.setattr(harness, "create_deep_agent", original)
    resumed = asyncio.run(harness.run_research(tmp_path, resume=True))
    assert resumed["status"] == "complete", resumed
    assert resumed["budget"]["deadline_at"] == deadline
    assert resumed["budget"]["search_rounds"] == 1
    assert resumed["budget"]["tool_calls"] == 4


def test_small_budget_returns_partial_report(tmp_path, task, corpus_path):
    config = BudgetConfig(max_model_calls=2, reserve_model_calls=1)
    result = asyncio.run(harness.run_research(tmp_path, task, corpus_path, config, mock=True))
    assert result["status"] == "partial", result
    assert result["budget"]["model_calls"] <= 2
    assert (tmp_path / "report.md").exists()


def test_deadline_cancels_model_and_preserves_accounting(tmp_path, task, corpus_path, monkeypatch):
    class SlowModel(harness.FixtureChatModel):
        async def _agenerate(self, *args, **kwargs):
            await asyncio.sleep(2)
            return await super()._agenerate(*args, **kwargs)

    monkeypatch.setattr(harness, "FixtureChatModel", SlowModel)
    # Allow SDK setup to finish so this exercises cancellation of a dispatched
    # request, rather than expiry before the first callback reserves usage.
    config = BudgetConfig(deadline_seconds=1, reserve_seconds=0)
    result = asyncio.run(harness.run_research(tmp_path, task, corpus_path, config, mock=True))
    assert result["status"] == "partial", result
    assert result["stop_reason"] == "deadline_reached"
    assert result["budget"]["unknown_usage_calls"] >= 1


def test_hidden_filesystem_tool_cannot_bypass_adapter(tmp_path, task, corpus_path, monkeypatch):
    class AdversarialModel(harness.FixtureChatModel):
        attempted: bool = False

        async def _agenerate(self, messages, **kwargs):
            if not self.attempted:
                from langchain_core.outputs import ChatGeneration, ChatResult

                self.attempted = True
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "read_file",
                                        "args": {"file_path": "/data/evaluation_only/gold"},
                                        "id": "forbidden-read",
                                    }
                                ],
                                usage_metadata={
                                    "input_tokens": 10,
                                    "output_tokens": 10,
                                    "total_tokens": 20,
                                },
                            )
                        )
                    ]
                )
            return await super()._agenerate(messages, **kwargs)

    monkeypatch.setattr(harness, "FixtureChatModel", AdversarialModel)
    result = asyncio.run(
        harness.run_research(tmp_path, task, corpus_path, BudgetConfig(), mock=True)
    )
    assert result["status"] == "complete", result
    events = [json.loads(s) for s in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "tool_denied" and e["name"] == "read_file" for e in events)
    for event in events:
        if event["kind"] == "model_surface":
            assert set(event["tools"]) <= {
                "search",
                "read_document",
                "record_claim",
                "get_passage",
                "list_evidence",
                "write_todos",
            }


def test_sdk_summarization_calls_are_metered(tmp_path, monkeypatch):
    model = harness.FixtureChatModel()
    monkeypatch.setattr(harness, "FixtureChatModel", lambda: model)
    original = harness.create_deep_agent

    def with_forced_summary(**kwargs):
        kwargs["middleware"] = [
            *kwargs["middleware"],
            SummarizationMiddleware(model=model, trigger=("messages", 4), keep=("messages", 2)),
        ]
        return original(**kwargs)

    monkeypatch.setattr(harness, "create_deep_agent", with_forced_summary)
    from news_agent.budget import Budget, BudgetCallback
    from news_agent.storage import RunStore

    async def scenario():
        store = RunStore(tmp_path)
        budget = Budget(BudgetConfig(), store)
        callback = BudgetCallback(budget, 0)
        graph = with_forced_summary(model=model, tools=[], middleware=[])
        await graph.ainvoke(
            {"messages": [HumanMessage(content=f"fixture context {i}") for i in range(8)]},
            {"callbacks": [callback]},
        )
        return budget

    budget = asyncio.run(scenario())
    assert model.auxiliary_calls >= 1
    assert budget.state["model_calls"] == model.calls
    assert budget.state["unknown_usage_calls"] == 0


def test_completed_tool_result_is_replayed_without_dispatch(tmp_path):
    from types import SimpleNamespace

    from langchain_core.messages import ToolMessage

    from news_agent.budget import Budget
    from news_agent.storage import RunStore

    async def scenario():
        store = RunStore(tmp_path)
        budget = Budget(BudgetConfig(), store)
        request = SimpleNamespace(tool_call={"name": "search", "args": {}, "id": "stable-id"})
        count = 0

        async def handler(_):
            nonlocal count
            count += 1
            return ToolMessage(content='{"results":[]}', tool_call_id="stable-id", name="search")

        first = harness.ResearchBoundary(budget, store, {"search"})
        await first.awrap_tool_call(request, handler)
        recovered = harness.ResearchBoundary(budget, store, {"search"})
        await recovered.awrap_tool_call(request, handler)
        return count

    assert asyncio.run(scenario()) == 1


def test_repeated_searches_cannot_strand_unread_candidates(tmp_path, task, corpus_path):
    from news_agent.budget import Budget
    from news_agent.retrieval import LocalCorpus
    from news_agent.storage import EvidenceStore, RunStore

    async def scenario():
        store = RunStore(tmp_path)
        evidence = EvidenceStore(store)
        budget = Budget(BudgetConfig(), store)
        tools = {
            t.name: t
            for t in harness.make_tools(LocalCorpus.load(corpus_path, task), evidence, budget)
        }
        # Simulate three search requests in one tool batch, before the next model turn.
        first = json.loads(await tools["search"].ainvoke({"query": task.event}))
        for _ in range(2):
            await tools["search"].ainvoke({"query": task.event})
        assert budget.state["phase"] == "research"
        assert not evidence.documents
        for hit in first["results"]:
            result = json.loads(
                await tools["read_document"].ainvoke({"version_id": hit["version_id"]})
            )
            assert result["passages"]
        await tools["search"].ainvoke({"query": task.event})
        assert budget.state["phase"] == "writing"
        assert budget.state["stop_reason"] == "retrieval_no_new_versions"
        denied = json.loads(await tools["search"].ainvoke({"query": task.event}))
        assert denied["error"] == "research_finished"

    asyncio.run(scenario())
