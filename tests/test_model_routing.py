"""Actual SDK + HTTP requests verify profiles, shared accounting and thinking isolation."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage
from openai import APIConnectionError, AuthenticationError
from test_adaptive_http import http_scenario

from news_agent import model_retry
from news_agent.budget import Budget, BudgetCallback
from news_agent.harness import run_research
from news_agent.model_routing import PROFILE, policy, stage_for
from news_agent.schemas import BudgetConfig
from news_agent.storage import RunStore, read_json
from news_agent.web_view import build_snapshot


@pytest.mark.parametrize(
    "names,writing,expected",
    [
        (["begin_investigation"], False, "scope_planning"),
        (["search_gap"], False, "gap_planning"),
        (["read_document", "open_url", "close_gap_round"], False, "retrieval_selection"),
        (["append_event_finding", "read_document"], False, "evidence_assessment"),
        (["resolve_event_write"], True, "evidence_repair"),
        ([], True, "report_draft"),
    ],
)
def test_route_uses_available_state_not_future_model_choice(names, writing, expected):
    assert stage_for([SimpleNamespace(name=n) for n in names], writing) == expected


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_staged_http_shared_budget_and_private_thinking_isolation(tmp_path, monkeypatch, finish):
    result, requests, web = asyncio.run(
        http_scenario(
            tmp_path,
            monkeypatch,
            model_profile=PROFILE,
            thinking_finish=finish,
        )
    )
    events = [json.loads(s) for s in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    routes = [e for e in events if e["kind"] == "model_route"]
    configs = [e for e in events if e["kind"] == "provider_request_config"]
    assert len(routes) == len(configs) == len(requests) == result["budget"]["model_calls"] == 11
    assert result["budget"]["tokens"] == 3850  # reasoning already included in completion usage
    assert len(web) == result["budget"]["retrieval_api_calls"] == 4
    assert {r["model"] for r in requests} == {"deepseek-v4-pro", "deepseek-flash"}
    for route, request, config in zip(routes, requests, configs, strict=True):
        assert request["model"] == route["model"].split(":", 1)[1]
        assert request["thinking"]["type"] == route["thinking"]
        assert request["max_tokens"] == 1500
        assert config["thinking"] == request["thinking"]
        assert route["reasoning_effort"] == request.get("reasoning_effort")
        assert config["reasoning_effort"] == request.get("reasoning_effort")
    assert routes[-1]["stage"] == "report_review"
    assert sum(r["thinking"]["type"] == "enabled" for r in requests) == 2
    revision = read_json(tmp_path / "report_revision.json")
    assert revision["status"] == ("applied" if finish == "stop" else "safe_partial")
    assert revision["model_calls"] == 1  # time audit now precedes final revision
    assert read_json(tmp_path / "temporal_review.json")["model_calls"] == 1
    assert read_json(tmp_path / "effective_config.json")["model_routing_policy"] == policy()
    view = build_snapshot(tmp_path)
    assert view["model_routing"] == policy() and len(view["model_routes"]) == 11
    for p in tmp_path.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".txt", ".md", ".html"}:
            assert "PRIVATE_REASONING_TEST_CANARY" not in p.read_text(), p
    with pytest.raises(ValueError, match="model profile changed"):
        asyncio.run(run_research(tmp_path, resume=True, model_profile="single"))


def test_report_draft_retries_after_transport_error_without_restarting_the_agent(
    tmp_path, monkeypatch
):
    waits = []

    async def observe_wait(seconds):
        waits.append(seconds)
        node = build_snapshot(tmp_path, active=True)["model_retries"][-1]
        assert (
            node["status"] == "waiting" and node["details"]["arguments"]["delay_seconds"] == seconds
        )
        assert "ReadError" in node["details"]["arguments"]["cause_types"]

    monkeypatch.setattr(model_retry, "wait_before_retry", observe_wait)
    config = BudgetConfig(
        max_tokens=None, max_model_calls=None, max_tool_calls=None, deadline_seconds=None
    )
    result, requests, web = asyncio.run(
        http_scenario(
            tmp_path,
            monkeypatch,
            model_profile=PROFILE,
            report_disconnect=True,
            budget_override=config,
        )
    )
    assert result["model_finalized"] and result["report_revision"]["status"] == "applied"
    assert waits == [12] and len(web) == 4
    assert result["budget"]["model_calls"] == len(requests) == 12
    assert result["budget"]["unknown_usage_calls"] == 1
    (retry,) = build_snapshot(tmp_path)["model_retries"]
    assert retry["final_status"] == "completed"
    assert retry["sequence"] < retry["started_sequence"] < retry["end_sequence"]
    events = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    dispatched = [e for e in events if e["kind"] == "model_dispatch"]
    assert len({e["action_id"] for e in dispatched}) == 12
    previous = next(
        i
        for i, e in enumerate(dispatched)
        if e["action_id"] == retry["details"]["arguments"]["retry_id"]
    )
    assert requests[previous] == requests[previous - 1]


@pytest.mark.parametrize(
    "failure,limit,expected_waits,error",
    [
        ("connect", None, [12, 16, 20], APIConnectionError),
        ("connect", 1, [], APIConnectionError),
        ("auth", None, [], AuthenticationError),
    ],
)
def test_deepseek_retry_stops_at_limit_and_excludes_auth_errors(
    tmp_path, monkeypatch, failure, limit, expected_waits, error
):
    waits = []

    async def wait(seconds):
        waits.append(seconds)

    def respond(request):
        if failure == "connect":
            raise httpx.ConnectError("transport unavailable", request=request)
        return httpx.Response(401, json={"error": {"message": "Invalid key"}})

    monkeypatch.setattr(model_retry, "wait_before_retry", wait)
    budget = Budget(BudgetConfig(max_model_calls=limit, reserve_model_calls=0), RunStore(tmp_path))
    budget.finish_research("fixture_writing")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model = model_retry.init_chat_model(
                "deepseek:deepseek-flash", api_key="dummy", http_async_client=client, timeout=60
            )
            with pytest.raises(error):
                await model.ainvoke(
                    [HumanMessage("Write a report")],
                    config={"callbacks": [BudgetCallback(budget, 0)]},
                )

    asyncio.run(scenario())
    assert waits == expected_waits
    assert budget.state["model_calls"] == 1 + len(waits)
    assert not budget.state["pending"]


def test_sdk_rejects_hallucinated_todo_without_reexecuting_and_keeps_investigation(
    tmp_path, monkeypatch
):
    result, requests, _ = asyncio.run(
        http_scenario(tmp_path, monkeypatch, model_profile=PROFILE, hidden_todo=True)
    )
    events = [json.loads(s) for s in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert result["model_finalized"]
    assert len(requests) == result["budget"]["model_calls"] == 12
    assert not any(e["kind"] == "tool_dispatch" and e["name"] == "write_todos" for e in events)
    assert any(e["kind"] == "adaptive_surface_denied" for e in events)
    assert any("do not call write_todos" in m.get("content", "") for m in requests[4]["messages"])
    assert all(
        "write_todos" not in {t["function"]["name"] for t in r.get("tools", [])} for r in requests
    )
