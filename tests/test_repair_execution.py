import asyncio
import copy
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_answer_contract import observation, task_at
from test_question_state import bind, deliver, setup
from test_repair_admission import partial

import news_agent.harness as harness
from news_agent.harness import make_tools
from news_agent.mock_model import FixtureChatModel
from news_agent.progress import document_locator, visual_search_tool
from news_agent.question_state import state_tool
from news_agent.repair_execution import RepairExecutor
from news_agent.retrieval import LocalCorpus
from news_agent.schemas import BudgetConfig, DocumentVersion
from news_agent.storage import read_json, write_json
from news_agent.web_view import build_snapshot


class RepairScenario(FixtureChatModel):
    """Real SDK scripted fixture; no provider calls or quality measurement."""

    mode: str = "success"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        content, actions = "", []
        is_stage = messages[-1].text.startswith("C3 bounded evidence step.")
        packet = json.loads(messages[-1].text.split("\n", 1)[1]) if is_stage else None
        if messages[0].text.startswith("REPORT_REVISION:"):
            originals = json.loads(messages[-1].text)["eligible_evidence"]["documents"]
            selected = next((d for d in originals if d["version_id"] == "closing"), originals[0])
            pid, quote = next(iter(selected["excerpts"].items()))
            closing = selected["version_id"] == "closing"
            content = json.dumps(
                {
                    "observations": [
                        observation(
                            quote,
                            passage_id=pid,
                            kind="completion" if closing else "approval",
                            target_event=closing,
                            subject="Aurora",
                        )
                    ]
                }
            )
        elif is_stage:
            stage = packet["repair_stage"]
            assert kwargs["tool_choice"]["function"]["name"] == (
                "search"
                if stage == "search"
                else "read_document"
                if stage == "read"
                else "find_saved_text"
                if stage == "locate"
                else "update_question_evidence"
            )
            if stage == "search":
                actions = [
                    (
                        "search",
                        {
                            "query": "unfindable"
                            if self.mode == "empty"
                            else "Closing Aurora Birch",
                            "purpose": "Find observed completion and date, not approval",
                            "intent": "gap",
                            "step_id": "main",
                            "limit": 1,
                        },
                    )
                ]
            elif stage == "read":
                actions = [("read_document", {"version_id": packet["candidates"][0]["version_id"]})]
            else:
                docs = packet["documents"]
                d = next((d for d in docs if d["version_id"] == "closing"), docs[0])
                pid, quote = next(iter(d["excerpts"].items()))
                closing = d["version_id"] == "closing"
                obj = observation(
                    quote,
                    passage_id=pid,
                    kind="completion" if closing else "approval",
                    target_event=closing,
                    subject="Aurora",
                )
                actions = [
                    (
                        "update_question_evidence",
                        {
                            "question_id": "main",
                            "seen_revision": packet["question_state"][0]["evidence_revision"],
                            "observation": obj,
                        },
                    )
                ]
        elif self.calls == 1:
            actions = [
                (
                    "search",
                    {
                        "query": "Consent Aurora Birch",
                        "step_id": "main",
                        "purpose": "Find initial official record",
                        "intent": "initial",
                        "limit": 1,
                    },
                )
            ]
        elif self.calls == 2:
            actions = [("read_document", {"version_id": "consent"})]
        else:
            content = "Answer from collected originals; unresolved evidence remains unknown."
        calls = [
            {"name": name, "args": args, "id": f"repair-{self.calls}-{i}", "type": "tool_call"}
            for i, (name, args) in enumerate(actions)
        ]
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=content,
                        tool_calls=calls,
                        response_metadata={"finish_reason": "tool_calls" if calls else "stop"},
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        },
                    )
                )
            ]
        )


def corpus(root):
    path = root / "corpus.json"
    write_json(
        path,
        [
            d.model_dump(mode="json")
            for d in [
                DocumentVersion(
                    version_id="consent",
                    title="Consent Aurora Birch",
                    url="https://regulator.example/consent",
                    content=(
                        "On October 13, 2023, the regulator gave Aurora consent to acquire Birch."
                    ),
                ),
                DocumentVersion(
                    version_id="closing",
                    title="Closing Aurora Birch",
                    url="https://issuer.example/closing",
                    content="On October 13, 2023, Aurora completed its acquisition of Birch.",
                ),
            ]
        ],
    )
    return path


@pytest.mark.parametrize("mode", ["success", "empty", "unavailable"])
def test_actual_sdk_agent_chooses_gap_query_reads_original_and_updates_same_question(
    tmp_path, monkeypatch, mode
):
    path = corpus(tmp_path)
    monkeypatch.setattr(harness, "FixtureChatModel", lambda: RepairScenario(mode=mode))
    if mode == "unavailable":
        original = LocalCorpus.read

        def fail_closing(self, vid):
            if vid == "closing":
                raise ValueError("original_source_unavailable")
            return original(self, vid)

        monkeypatch.setattr(LocalCorpus, "read", fail_closing)
    run = tmp_path / "run"
    result = asyncio.run(
        harness.run_research(
            run,
            task_at().model_copy(update={"synthetic": True}),
            path,
            BudgetConfig(),
            mock=True,
            visual_progress=True,
        )
    )
    assert result["model_finalized"], result
    assert result["report_revision"]["status"] == "applied", result
    state = read_json(run / "question_state.json")
    execution = read_json(run / "repair_execution.json")
    flow = execution["workflow"]
    assert flow["searches"] == 1 and flow["reads"] <= 1
    assert flow["status"] == ("covered_provisionally" if mode == "success" else "incomplete")
    assert state["questions"]["main"]["status"] == (
        "answered_provisionally" if mode == "success" else "unresolved"
    )
    answer = read_json(run / "answer_state.json")
    assert answer["inference"]["status"] == (
        "not_completed_by_cutoff" if mode == "success" else "unknown"
    )
    assert result["budget"]["search_rounds"] == 2
    assert result["budget"]["tokens"] == result["budget"]["model_calls"] * 120
    assert not result["budget"]["pending"]
    if mode == "success":
        assert flow["delivered"] and flow["assessments"] == 1
        snapshot = build_snapshot(run)
        assert snapshot["repair_execution"] == execution
        assert any(
            n.get("provenance") == "runtime_bounded_repair_policy" for n in snapshot["nodes"]
        )
        assert any(
            e["from"].startswith("repair-step:") and e["to"].startswith("tool:")
            for e in snapshot["edges"]
        )
        assert (
            read_json(run / "repair_before.json")["questions"]["main"]["status"]
            == "partial_support"
        )
        assert (
            read_json(run / "repair_after.json")["questions"]["main"]["status"]
            == "answered_provisionally"
        )


def executor(root):
    task, evidence, budget, questions = partial(root)
    path = corpus(root)
    source = LocalCorpus.load(path, task)
    tools = [
        visual_search_tool(t, evidence, budget) if t.name == "search" else t
        for t in make_tools(source, evidence, budget)
    ]
    tools += [state_tool(questions), document_locator(evidence, budget)]
    return evidence, budget, questions, tools, RepairExecutor(questions, budget)


def test_reserved_dispatch_rechecks_capacity_and_refuses_batched_extra_search(tmp_path):
    _, b, _, tools, repair = executor(tmp_path)
    surface = repair.prepare(tools, None)
    assert surface[2] == "search"
    call = {"id": "one", "name": "search", "args": {"step_id": "main", "intent": "gap"}}
    assert repair.before_tool(call) is None
    assert repair.before_tool({**call, "id": "two"}) == "one_tool_call_per_bounded_repair_turn"
    assert repair.data["workflow"]["searches"] == 1
    assert b.state["model_calls"] == 0  # scheduling does not fabricate a model charge


@pytest.mark.parametrize("cached", [False, True])
def test_resume_uses_cached_result_or_stops_unknown_tool_without_redispatch(tmp_path, cached):
    _, b, q, tools, repair = executor(tmp_path)
    repair.prepare(tools, None)
    call = {"id": "one", "name": "search", "args": {"step_id": "main", "intent": "gap"}}
    repair.before_tool(call)
    if cached:
        write_json(
            tmp_path / "tool_cache.json",
            {
                "one": {
                    "content": json.dumps(
                        {
                            "results": [
                                {
                                    "version_id": "closing",
                                    "title": "Closing",
                                    "url": "https://issuer.example/closing",
                                }
                            ]
                        }
                    )
                }
            },
        )
    before = copy.deepcopy(b.state)
    resumed = RepairExecutor(q, b)
    assert resumed.data["workflow"]["stage"] == ("read" if cached else "search")
    assert resumed.data["workflow"]["status"] == ("running" if cached else "incomplete")
    if cached:
        assert b.state == before
    else:
        assert b.state["phase"] == "writing" and b.state["model_calls"] == 0
    assert resumed.data["workflow"]["searches"] == 1


def test_covered_main_and_budget_shortfall_never_dispatch_repair(tmp_path):
    e, b, q, tools, repair = executor(tmp_path)
    b.state["tokens"] = 59999
    assert repair.prepare(tools, None) is None
    assert repair.data["workflow"]["status"] == "not_admitted"
    assert b.state["model_calls"] == 0 and b.state["search_rounds"] == 0
    other = tmp_path / "covered"
    _, e, b, q = setup(other)
    bind(q, deliver(e, b, q))
    assert RepairExecutor(q, b).prepare(tools, None) is None


def test_capacity_change_before_dispatch_stops_without_spending(tmp_path):
    _, b, _, tools, repair = executor(tmp_path)
    repair.prepare(tools, None)
    b.state["phase"] = "writing"
    error = repair.before_tool(
        {"id": "never", "name": "search", "args": {"step_id": "main", "intent": "gap"}}
    )
    assert error == "research_closed_preserve_writing_reserve"
    assert repair.data["workflow"]["searches"] == 0
    assert b.state["model_calls"] == 0


def test_unknown_model_usage_is_retained_and_resume_does_not_retry_stage(tmp_path):
    _, b, q, tools, repair = executor(tmp_path)
    repair.prepare(tools, None)
    b.before_model("uncertain-model-request", 1000)
    b.settle_pending()
    resumed = RepairExecutor(q, b, interrupted_model=True)
    assert resumed.data["workflow"]["reason"] == "interrupted_model_outcome_unknown"
    assert b.state["model_calls"] == 1 and b.state["tokens"] == 2500
    assert b.state["unknown_usage_calls"] == 1 and b.state["phase"] == "writing"
    assert resumed.prepare(tools, None) is None


def test_missing_execution_checkpoint_cannot_reset_reserved_attempt(tmp_path):
    _, b, q, tools, repair = executor(tmp_path)
    repair.prepare(tools, None)
    (tmp_path / "repair_execution.json").unlink()
    resumed = RepairExecutor(q, b)
    assert resumed.data["workflow"]["reason"] == "missing_execution_checkpoint"
    assert b.state["phase"] == "writing"
    assert resumed.prepare(tools, None) is None


def test_wrong_model_tool_cannot_loop_unboundedly_or_dispatch_network(tmp_path):
    _, b, _, tools, repair = executor(tmp_path)
    repair.prepare(tools, None)
    assert repair.before_tool({"id": "wrong", "name": "open_url", "args": {}})
    assert repair.prepare(tools, None) is None
    assert repair.data["workflow"]["reason"] == "model_did_not_execute_selected_step"
    assert repair.data["workflow"]["searches"] == 0 and b.state["model_calls"] == 0
