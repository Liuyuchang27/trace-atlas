import asyncio
import json

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import news_agent.harness as harness
from news_agent.budget import Budget, BudgetCallback
from news_agent.mechanisms import Investigation
from news_agent.retrieval import LocalCorpus
from news_agent.schemas import BudgetConfig
from news_agent.selections import excerpts
from news_agent.storage import EvidenceStore, RunStore, read_json


class ReviewModel(BaseChatModel):
    replies: list[dict]

    @property
    def _llm_type(self):
        return "scripted-review-fixture"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(
            content=json.dumps(self.replies.pop(0)),
            usage_metadata={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def setup_investigation(tmp_path, task, corpus_path, condition, replies):
    store = RunStore(tmp_path)
    evidence = EvidenceStore(store)
    budget = Budget(BudgetConfig(), store)
    corpus = LocalCorpus.load(corpus_path, task)
    for doc in corpus.documents.values():
        evidence.add_document(doc)
    pid = next(
        p.passage_id for p in evidence.passages.values() if p.version_id == "announcement-v1"
    )
    claim = evidence.add_claim(evidence.passages[pid].text, [pid], "2026-06-01", "announcement")
    state = Investigation(
        condition,
        task,
        evidence,
        budget,
        ReviewModel(replies=replies),
        harness.make_tools(corpus, evidence, budget),
    )
    return state, claim, pid, {"callbacks": [BudgetCallback(budget, 0)]}


@pytest.mark.parametrize("bad_quote", [False, True])
def test_support_review_is_metered_cached_and_does_not_certify_claims(
    tmp_path, task, corpus_path, bad_quote
):
    state, claim, pid, config = setup_investigation(tmp_path, task, corpus_path, "s", [])
    state.model.replies.append(
        {
            "links": [
                {
                    "passage_id": pid,
                    "relation": "supports",
                    "excerpt_id": "bad-id"
                    if bad_quote
                    else excerpts(pid, state.evidence.passages[pid].text)[0]["excerpt_id"],
                    "reason": "Scripted fixture judgment.",
                }
            ]
        }
    )
    tools = {t.name: t for t in state.tools()}
    result = json.loads(
        asyncio.run(tools["verify_claim"].ainvoke({"claim_id": claim.claim_id}, config))
    )
    assert state.budget.state["model_calls"] == state.state["review_model_calls"] == 1
    assert state.budget.state["tokens"] == 100
    assert state.evidence.claims[0].verification == "unverified"
    if bad_quote:
        assert result["error"] == "ValueError"
        assert not state.state["support"]
    else:
        assert result["assessment"]["recommended_citations"] == [pid]
        restored = Investigation(
            "s", task, state.evidence, state.budget, state.model, list(state.common.values())
        )
        tool = next(t for t in restored.tools() if t.name == "verify_claim")
        cached = json.loads(asyncio.run(tool.ainvoke({"claim_id": claim.claim_id}, config)))
        assert cached["assessment_status"] == "model_assessed"
        assert state.budget.state["model_calls"] == 1


@pytest.mark.parametrize("kind", ["planned", "observed"])
def test_temporal_check_preserves_future_plan_but_rejects_future_observation(
    tmp_path, task, corpus_path, kind
):
    state, claim, pid, config = setup_investigation(tmp_path, task, corpus_path, "t", [])
    state.model.replies.append(
        {
            "stage": "completion",
            "kind": kind,
            "event_date": "2026-06-20",
            "verdict": "supported",
            "passage_id": pid,
            "excerpt_id": excerpts(pid, state.evidence.passages[pid].text)[0]["excerpt_id"],
            "reason": "Scripted fixture judgment.",
        }
    )
    tool = next(t for t in state.tools() if t.name == "check_event_transition")
    result = json.loads(asyncio.run(tool.ainvoke({"claim_id": claim.claim_id}, config)))
    assert state.state["gaps"]["t-completion"]["status"] == "open"
    if kind == "planned":
        assert result["assessment"]["kind"] == "planned"
    else:
        assert result["error"] == "ValueError"
        assert not state.state["temporal"]


def test_unknown_origin_is_not_automatically_grouped_or_independent(tmp_path, task, corpus_path):
    state, _, _, config = setup_investigation(tmp_path, task, corpus_path, "s", [])
    state.model.replies.append(
        {
            "relation": "unknown",
            "confidence": "unknown",
            "evidence": [],
            "needs_independent_record": True,
            "reason": "Different domains alone are insufficient.",
        }
    )
    tool = next(t for t in state.tools() if t.name == "trace_origin")
    result = json.loads(
        asyncio.run(tool.ainvoke({"version_ids": ["announcement-v1", "approval-v1"]}, config))
    )
    assert result["assessment"]["relation"] == "unknown"
    assert len(state.state["gaps"]) == 1
    assert next(iter(state.state["gaps"].values()))["status"] == "open"


def test_gap_actions_use_common_retriever_and_persist_exhaustion(tmp_path, task, corpus_path):
    state, _, _, config = setup_investigation(tmp_path, task, corpus_path, "t", [])
    tools = {t.name: t for t in state.tools()}
    for _ in range(2):
        value = json.loads(
            asyncio.run(
                tools["focus_date_or_subevent"].ainvoke(
                    {"gap_id": "t-completion", "query": task.event}, config
                )
            )
        )
        assert not value["new_candidates"]
        assert all(r["version_id"] != "closing-after-cutoff" for r in value["results"])
    assert state.budget.state["tool_calls"] == 2
    assert state.state["gaps"]["t-completion"]["status"] == "unresolved"
    assert read_json(tmp_path / "investigation.json")["gaps"]["t-completion"]["attempts"] == 2


def test_window_expansion_requires_a_real_open_gap_and_keeps_cutoff_filter(
    tmp_path, task, corpus_path
):
    state, _, _, config = setup_investigation(tmp_path, task, corpus_path, "t", [])
    tool = next(t for t in state.tools() if t.name == "extend_time_window")
    old = dict(state.state["window"])
    invalid = json.loads(
        asyncio.run(
            tool.ainvoke(
                {
                    "gap_id": "missing",
                    "start": "2025-01-01",
                    "end": "2026-07-01",
                    "query": task.event,
                },
                config,
            )
        )
    )
    assert "error" in invalid and state.state["window"] == old
    result = json.loads(
        asyncio.run(
            tool.ainvoke(
                {
                    "gap_id": "t-completion",
                    "start": "2025-01-01",
                    "end": "2026-07-01",
                    "query": task.event,
                },
                config,
            )
        )
    )
    assert "error" not in result
    assert state.state["window"]["end"] == "2026-07-01"
    assert all(r["version_id"] != "closing-after-cutoff" for r in result["results"])


def test_research_budget_blocks_auxiliary_calls_and_bad_ids_are_not_reviewed(
    tmp_path, task, corpus_path
):
    state, claim, _, config = setup_investigation(tmp_path, task, corpus_path, "s", [])
    tool = next(t for t in state.tools() if t.name == "verify_claim")
    invalid = json.loads(asyncio.run(tool.ainvoke({"claim_id": "not-recorded"}, config)))
    assert invalid["error"] == "ValueError"
    assert state.budget.state["model_calls"] == 0
    state.budget.finish_research("research_budget_exhausted")
    blocked = json.loads(asyncio.run(tool.ainvoke({"claim_id": claim.claim_id}, config)))
    assert blocked["error"] == "research_finished"
    assert state.budget.state["model_calls"] == 0


def test_temporal_template_is_not_forced_on_unrelated_events(tmp_path, task, corpus_path):
    unrelated = task.model_copy(update={"event": "City election", "question": "Who won?"})
    with pytest.raises(ValueError, match="acquisition event template"):
        setup_investigation(tmp_path, unrelated, corpus_path, "t", [])


def test_pending_checks_defer_stagnation_and_are_interleaved(tmp_path, task, corpus_path):
    state, claim, _, _ = setup_investigation(tmp_path, task, corpus_path, "ts", [])
    assert state.required_tool() == "check_event_transition"
    assert state.allow_retrieval_stop() is False
    assert state.allow_retrieval_stop() is False
    assert state.allow_retrieval_stop() is True
    state.state["review_attempts"]["T"] = 1
    assert state.required_tool() == "verify_claim"
    state.state["review_attempts"]["S"] = 1
    assert state.required_tool() == "trace_origin"
    state.budget.finish_research("research_budget_exhausted")
    assert state.required_tool() is None
    assert state.recommended_actions() == []


@pytest.mark.parametrize("condition", ["baseline", "t", "s", "ts"])
def test_real_sdk_nested_deepseek_calls_are_isolated_and_counted_once(
    tmp_path, task, corpus_path, monkeypatch, condition
):
    requests, seen = [], {}
    outer = 0

    def respond(request):
        nonlocal outer
        body = json.loads(request.content)
        requests.append(body)
        serialized = json.dumps(body)
        assert "FUTURE_COMPLETION_SENTINEL" not in serialized
        assert "REVISED_CONTENT_SENTINEL" not in serialized
        system_content = body["messages"][0]["content"]
        system = (
            system_content
            if isinstance(system_content, str)
            else "".join(block.get("text", "") for block in system_content)
        )
        calls = []
        content = ""
        if system.startswith("REPORT_REVISION:"):
            assert not body.get("tools")
            content = json.loads(body["messages"][-1]["content"])["draft"]
        elif system.startswith("MECHANISM_REVIEW:"):
            assert not body.get("tools")
            assert body["response_format"] == {"type": "json_object"}
            payload = json.loads(body["messages"][-1]["content"])
            if system.startswith("MECHANISM_REVIEW:T"):
                assert condition in {"t", "ts"}
                p = payload["passages"][0]
                value = {
                    "stage": "announcement",
                    "kind": "observed",
                    "event_date": "2026-06-01",
                    "verdict": "supported",
                    "passage_id": p["passage_id"],
                    "excerpt_id": p["excerpts"][0]["excerpt_id"],
                    "reason": "HTTP mock fixture.",
                }
            elif system.startswith("MECHANISM_REVIEW:S"):
                assert condition in {"s", "ts"}
                value = {
                    "links": [
                        {
                            "passage_id": p["passage_id"],
                            "relation": "supports",
                            "excerpt_id": p["excerpts"][0]["excerpt_id"],
                            "reason": "HTTP mock fixture.",
                        }
                        for p in payload["passages"]
                    ]
                }
            else:
                assert condition in {"s", "ts"}
                value = {
                    "relation": "unknown",
                    "confidence": "unknown",
                    "evidence": [],
                    "needs_independent_record": False,
                    "reason": "HTTP mock fixture.",
                }
            content = json.dumps(value)
        else:
            outer += 1
            names = {t["function"]["name"] for t in body.get("tools", [])}
            if condition not in {"t", "ts"}:
                assert "check_event_transition" not in names
            if condition not in {"s", "ts"}:
                assert "verify_claim" not in names and "trace_origin" not in names
            if isinstance(body.get("tool_choice"), dict):
                assert names == {body["tool_choice"]["function"]["name"]}
            elif outer == 1:
                assert ("check_event_transition" in names) == (condition in {"t", "ts"})
                assert ("verify_claim" in names) == (condition in {"s", "ts"})
            if outer == 1:
                calls = [("search", {"query": task.event})]
            elif outer == 2:
                calls = [
                    ("read_document", {"version_id": vid})
                    for vid in ["announcement-v1", "approval-v1", "republication-v1"]
                ]
            elif outer == 3:
                original = next(
                    json.loads(m["content"])
                    for m in body["messages"]
                    if m["role"] == "tool" and "passages" in json.loads(m["content"])
                )
                p = original["passages"][0]
                seen["pid"] = p["passage_id"]
                calls = [
                    (
                        "record_claim",
                        {
                            "text": p["text"],
                            "passage_ids": [p["passage_id"]],
                            "event_time": "2026-06-01",
                            "stage": "announcement",
                        },
                    )
                ]
            elif condition != "baseline" and isinstance(body.get("tool_choice"), dict):
                name = body["tool_choice"]["function"]["name"]
                original_claim = next(
                    json.loads(m["content"])["claim"]
                    for m in body["messages"]
                    if m["role"] == "tool" and "claim" in json.loads(m["content"])
                )
                arguments = {"version_ids": ["announcement-v1", "approval-v1"]}
                if name != "trace_origin":
                    arguments = {"claim_id": original_claim["claim_id"]}
                calls = [(name, arguments)]
            else:
                content = f"HTTP mock report only [{seen['pid']}]."
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = [
                {
                    "id": f"outer-{outer}-{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
                for i, (name, args) in enumerate(calls)
            ]
        return httpx.Response(
            200,
            json={
                "id": f"mock-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            },
        )

    original = harness.init_chat_model

    async def scenario():
        transport = httpx.MockTransport(respond)
        with httpx.Client(transport=transport) as sync:
            async with httpx.AsyncClient(transport=transport) as async_client:
                monkeypatch.setattr(
                    harness,
                    "init_chat_model",
                    lambda *a, **k: original(
                        *a,
                        api_key="test-not-a-real-key",
                        http_client=sync,
                        http_async_client=async_client,
                        **k,
                    ),
                )
                return await harness.run_research(
                    tmp_path,
                    task,
                    corpus_path,
                    BudgetConfig(),
                    model_name="deepseek:deepseek-flash",
                    condition=condition,
                )

    result = asyncio.run(scenario())
    assert result["status"] == "complete", result
    assert result["condition"] == condition
    assert result["budget"]["model_calls"] == len(requests)
    assert result["budget"]["tokens"] == 130 * len(requests)
    assert result["budget"]["unknown_usage_calls"] == 0
    auxiliary = (condition in {"t", "ts"}) + 2 * (condition in {"s", "ts"})
    if condition == "baseline":
        assert not (tmp_path / "investigation.json").exists()
    else:
        assert result["mechanisms"]["review_model_calls"] == auxiliary
        assert result["mechanisms"]["actions"] == auxiliary
        assert all(c["verification"] == "unverified" for c in read_json(tmp_path / "claims.json"))
    before = len(requests)
    asyncio.run(harness.run_research(tmp_path, resume=True))
    assert len(requests) == before


def test_failed_target_is_not_retried_after_resume_or_exposed_again(tmp_path, task, corpus_path):
    state, claim, pid, config = setup_investigation(tmp_path, task, corpus_path, "s", [])
    state.model.replies.append(
        {
            "links": [
                {
                    "passage_id": pid,
                    "relation": "supports",
                    "excerpt_id": "not-supplied",
                    "reason": "Fixture.",
                }
            ]
        }
    )
    tool = next(t for t in state.tools() if t.name == "verify_claim")
    first = json.loads(asyncio.run(tool.ainvoke({"claim_id": claim.claim_id}, config)))
    assert first["error_code"] == "excerpt_not_in_supplied_source"
    used = state.budget.state["tool_calls"]
    restored = Investigation(
        "s", task, state.evidence, state.budget, state.model, list(state.common.values())
    )
    assert "verify_claim" not in restored.available_tools()
    again = json.loads(
        asyncio.run(
            next(t for t in restored.tools() if t.name == "verify_claim").ainvoke(
                {"claim_id": claim.claim_id}, config
            )
        )
    )
    assert again["error_code"] == "target_already_attempted"
    assert restored.budget.state["model_calls"] == 1
    assert restored.budget.state["tool_calls"] == used


def test_status_claim_beats_recent_price_and_origin_selection_targets_duplicates(
    tmp_path, task, corpus_path
):
    state, _, pid, _ = setup_investigation(tmp_path, task, corpus_path, "ts", [])
    core = state.evidence.add_claim(
        "Completion is not documented; the closing condition remains pending.",
        [pid],
        None,
        "completion",
    )
    price = state.evidence.add_claim(
        "The purchase price was 95 dollars per share.", [pid], None, None
    )
    assert state.pending_claims("T")[0].claim_id == core.claim_id
    assert state.pending_claims("S")[0].claim_id == core.claim_id
    assert state.claim_priority(core)[0] > state.claim_priority(price)[0]
    pair = state.origin_pair()
    assert set(pair) == {"approval-v1", "republication-v1"}
    assert not state.state["origins"]  # priority is never an origin verdict
