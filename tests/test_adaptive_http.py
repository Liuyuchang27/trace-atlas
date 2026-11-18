"""Real SDK/DeepSeek/Tavily under HTTP mocks, including unavailable originals."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, convert_to_messages
from test_adaptive_sdk import AdaptiveScenario, ConsecutiveWritingScenario, WritingRecoveryScenario

import news_agent.harness as harness
import news_agent.model_routing as routing
import news_agent.retrieval as retrieval
from news_agent.schemas import BudgetConfig, ResearchTask
from news_agent.storage import read_json


async def http_scenario(
    run,
    monkeypatch,
    fail=False,
    writing_recovery=False,
    consecutive=None,
    yearless=False,
    date_recovery=False,
    usage_profile=None,
    task_override=None,
    budget_override=None,
    model_profile=None,
    thinking_finish="stop",
    hidden_todo=False,
    report_disconnect=False,
):
    urls = {k: f"https://issuer.example/{k}" for k in ["early", "later"]}
    versions = {
        k: "web-" + hashlib.sha256(url.encode()).hexdigest()[:20] for k, url in urls.items()
    }
    scenario = (
        ConsecutiveWritingScenario(versions=versions, recovery_outcome=consecutive)
        if consecutive
        else (WritingRecoveryScenario if writing_recovery else AdaptiveScenario)(
            versions=versions, source_failure=fail
        )
    )
    scenario.yearless = yearless
    scenario.date_recovery = date_recovery
    if date_recovery:
        scenario.automatic_round_close = True
    models, web = [], []
    disconnected = False
    monkeypatch.setenv("TAVILY_API_KEY", "p2-http-dummy-web-key")
    holder = {}
    if writing_recovery or consecutive or yearless:
        original_budget = harness.Budget

        def capture_budget(*args, **kwargs):
            holder["budget"] = original_budget(*args, **kwargs)
            return holder["budget"]

        monkeypatch.setattr(harness, "Budget", capture_budget)

    def respond(request):
        nonlocal disconnected
        body = json.loads(request.content)
        if request.url.host == "api.tavily.com":
            web.append((request.url.path, body))
            if request.url.path == "/search":
                vid = "later" if "correction" in body["query"] else "early"
                assert "event date focus:" in body["query"]
                return httpx.Response(
                    200,
                    json={
                        "results": [{"url": urls[vid], "title": "Aurora statement"}],
                        "usage": {"credits": 1},
                    },
                )
            assert request.url.path == "/extract"
            later = body["urls"] == [urls["later"]]
            if later and fail:
                return httpx.Response(503, json={"error": "original unavailable"})
            text = (
                "On November 2, 2026, Aurora corrected its statement: "
                "only some batches were included."
                if later
                else "On November 1, 2026, Aurora said all batches were included."
            )
            if yearless:
                text = text.replace(", 2026,", ",")
                text = (
                    "Published on 2026-11-03\n\n"
                    + "\n\n".join(
                        f"[share {i}](https://issuer.example/share/{i})" for i in range(5)
                    )
                    + "\n\n"
                    + text
                )
            if later and (writing_recovery or consecutive or yearless):
                holder["budget"].finish_research("fixture_exhausted_after_original")
            return httpx.Response(
                200,
                json={
                    "results": [{"url": body["urls"][0], "raw_content": text}],
                    "usage": {"credits": 0.2},
                },
            )
        assert request.url.host == "api.deepseek.com"
        models.append(body)
        if (
            report_disconnect
            and not disconnected
            and any(
                "The research allowance has ended." in str(message["content"])
                for message in body["messages"]
            )
        ):
            disconnected = True
            assert request.extensions["timeout"]["connect"] == 15
            raise httpx.ReadError(
                "simulated disconnect after sending draft request", request=request
            )
        injected = hidden_todo and len(models) == 4
        if injected:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "hidden-todo",
                        "name": "write_todos",
                        "args": {"todos": []},
                    }
                ],
            )
        else:
            message = (
                scenario._generate(
                    convert_to_messages(body["messages"]), tools=body.get("tools", [])
                )
                .generations[0]
                .message
            )
        if body.get("tool_choice") == "required":
            assert message.tool_calls, "HTTP fixture must honor the forced-tool constraint"
        available = {t["function"]["name"] for t in body.get("tools", [])}
        assert injected or all(c["name"] in available for c in message.tool_calls)
        calls = [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
            }
            for c in message.tool_calls
        ]
        value = {"role": "assistant", "content": message.text}
        thinking = body.get("thinking", {}).get("type") == "enabled"
        if thinking:
            assert not body.get("tools")
            assert all(m["role"] in {"system", "user"} for m in body["messages"])
            assert body["reasoning_effort"] == "high"
            value["reasoning_content"] = "PRIVATE_REASONING_TEST_CANARY"
        if calls:
            value["tool_calls"] = calls
        return httpx.Response(
            200,
            json={
                "id": f"p2-http-{len(models)}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": value,
                        "finish_reason": "tool_calls"
                        if calls
                        else thinking_finish
                        if thinking
                        else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": (usage_profile[len(models) - 1] - 50)
                    if usage_profile
                    else 300,
                    "completion_tokens": 50,
                    "completion_tokens_details": {"reasoning_tokens": 30 if thinking else 0},
                    "total_tokens": usage_profile[len(models) - 1] if usage_profile else 350,
                },
            },
        )

    transport = httpx.MockTransport(respond)
    original_init, original_async = harness.init_chat_model, httpx.AsyncClient
    with httpx.Client(transport=transport) as sync_client:
        async with original_async(transport=transport) as async_client:
            model = original_init(
                "deepseek:deepseek-flash",
                api_key="p2-http-dummy-model-key",
                http_client=sync_client,
                http_async_client=async_client,
                max_tokens=1500,
                max_retries=0,
                temperature=0,
                timeout=60,
                extra_body={"thinking": {"type": "disabled"}},
            )
            monkeypatch.setattr(
                retrieval,
                "httpx",
                SimpleNamespace(
                    AsyncClient=lambda **kwargs: original_async(transport=transport, **kwargs)
                ),
            )
            monkeypatch.setattr(harness, "init_chat_model", lambda *args, **kwargs: model)
            monkeypatch.setattr(
                routing,
                "init_chat_model",
                lambda name, **kw: original_init(
                    name,
                    api_key="dummy",
                    http_client=sync_client,
                    http_async_client=async_client,
                    **kw,
                ),
            )
            result = await harness.run_research(
                run,
                task_override
                or ResearchTask(
                    task_id="p2-http",
                    event="Aurora",
                    question="Given the later result, how did the statements change?",
                    mode="retrospective",
                    synthetic=True,
                ),
                budget_config=budget_override
                or (BudgetConfig(max_model_calls=13) if fail else BudgetConfig()),
                adapter="tavily",
                model_name="deepseek:deepseek-flash",
                workflow="adaptive-event-map",
                model_profile=model_profile,
            )
    return result, models, web


def test_date_refusal_http_forces_one_explicit_unknown_date_decision_in_original_turns(
    tmp_path, monkeypatch
):
    result, models, web = asyncio.run(
        http_scenario(tmp_path, monkeypatch, yearless=True, date_recovery=True)
    )
    assert result["model_finalized"]
    assert len(models) == result["budget"]["model_calls"] == 11
    assert result["budget"]["writing_model_calls"] == 3
    assert result["budget"]["tokens"] == 3850
    assert len(web) == 4
    saved = read_json(tmp_path / "event_write_recovery.json")["proposal"]
    assert saved["kind"] == "date_support" and saved["status"] == "recovered"
    assert saved["model_opportunities"] == 1
    assert saved["finding"]["event_time"]["start"] == "2026-11-02"
    assert saved["resolution_finding"]["event_time"]["precision"] == "unknown"
    assert models[7]["tool_choice"]["function"]["name"] == "resolve_event_write"
    assert {t["function"]["name"] for t in models[7]["tools"]} == {"resolve_event_write"}
    graph = read_json(tmp_path / "event_graph.json")
    assert (
        graph["nodes"]["premise"]["findings"][-1]["finding"]["event_time"]["precision"] == "unknown"
    )


def test_yearless_http_binds_both_originals_without_event_year_inference(tmp_path, monkeypatch):
    result, models, web = asyncio.run(http_scenario(tmp_path, monkeypatch, yearless=True))
    assert result["model_finalized"]
    assert len(models) == result["budget"]["model_calls"] == 10
    assert len(web) == 4
    graph = read_json(tmp_path / "event_graph.json")
    rows = graph["nodes"]["premise"]["findings"][1:]
    assert len(rows) == 2
    assert all(r["finding"]["event_time"]["precision"] == "unknown" for r in rows)
    assert all("Original date text:" in r["finding"]["uncertainty"] for r in rows)
    calls = [json.loads(s) for s in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert (
        sum(e["kind"] == "tool_dispatch" and e["name"] == "append_undated_finding" for e in calls)
        == 2
    )
    evidence = read_json(tmp_path / "evidence.json")
    assert any("Published on 2026-11-03" in p["text"] for p in evidence["passages"].values())


@pytest.mark.parametrize("fail", [False, True])
def test_real_http_two_rounds_are_metered_once_and_source_failure_stays_unresolved(
    tmp_path, monkeypatch, fail
):
    result, models, web = asyncio.run(http_scenario(tmp_path, monkeypatch, fail))
    assert result["model_finalized"], result
    assert len(models) == result["budget"]["model_calls"] == (10 if fail else 11)
    assert result["budget"]["tokens"] == len(models) * 350
    assert len(web) == result["budget"]["retrieval_api_calls"] == 4
    assert result["budget"]["web_search_requests"] == 2
    assert result["budget"].get("retrieval_unknown_usage_calls", 0) == int(fail)
    state = read_json(tmp_path / "adaptive_investigation.json")
    assert len(state["rounds"]) == 2
    assert state["rounds"][1]["evidence_gain"] == (not fail)
    unresolved = next(
        g
        for g in state["gaps"].values()
        if g["node_id"] == "premise" and g["kind"] == "question_or_explanation_unresolved"
    )
    assert unresolved["status"] == ("open" if fail else "resolved")
    assert not result["budget"]["pending"] and not result["budget"].get("retrieval_pending")
    assert all(
        m.get("extra_body", {}).get("thinking", {"type": "disabled"})["type"] == "disabled"
        for m in models
    )


def test_one_bounded_writing_correction_binds_fresh_original_without_extra_research(
    tmp_path, monkeypatch
):
    result, models, web = asyncio.run(http_scenario(tmp_path, monkeypatch, writing_recovery=True))
    assert result["model_finalized"] and len(models) == result["budget"]["model_calls"] == 11
    assert result["budget"]["model_calls"] <= BudgetConfig().max_model_calls
    assert result["budget"]["tokens"] <= BudgetConfig().max_tokens
    assert len(web) == 4 and result["budget"]["web_search_requests"] == 2
    events = [json.loads(s) for s in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    denied = [e for e in events if e["kind"] == "adaptive_tool_denied"]
    assert not denied
    assert any(e["kind"] == "adaptive_prior_evidence_binding" for e in events)
    graph = read_json(tmp_path / "event_graph.json")
    assert graph["nodes"]["premise"]["findings"][-1]["finding"]["answers_node_question"]
    assert "append_event_finding" in {t["function"]["name"] for t in models[7]["tools"]}
    assert not models[8].get("tools")
