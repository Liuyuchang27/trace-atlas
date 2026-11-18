import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

import news_agent.harness as harness
import news_agent.retrieval as retrieval
from news_agent.live import question_task
from news_agent.schemas import BudgetConfig
from news_agent.storage import read_json


def test_live_question_has_observation_clock_without_seed_material():
    task = question_task("项目 Aurora 完成了吗？")
    assert task.mode == "retrospective" and task.synthetic is False
    question, clock = task.question.split("\n\nRun observation context: ")
    assert question == task.event == "项目 Aurora 完成了吗？"
    clock = json.loads(clock)
    utc, local = map(datetime.fromisoformat, [clock["utc"], clock["local"]])
    assert utc == local and local.utcoffset() == timedelta(hours=8)
    assert "http" not in task.model_dump_json()
    with pytest.raises(ValueError, match="nonempty"):
        question_task(" ")
    with pytest.raises(ValueError, match="timezone"):
        question_task("question", timezone="invalid/no-such-zone")


def test_missing_live_key_fails_before_model_or_network(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("must reject missing key before constructing a paid model")

    monkeypatch.setattr(harness, "init_chat_model", forbidden)
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        asyncio.run(
            harness.run_research(
                tmp_path,
                question_task("test question"),
                budget_config=BudgetConfig(),
                adapter="tavily",
                model_name="deepseek:deepseek-flash",
            )
        )
    assert not (tmp_path / "effective_config.json").exists()


def test_model_controls_query_selection_followup_and_link_read(tmp_path, monkeypatch):
    """Scripted HTTP acceptance of the real SDK loop, not semantic evaluation."""
    model_requests, web_requests, passages = [], [], []
    first = "https://issuer.example/aurora"
    linked = "https://records.example/decision-84"
    secret = "test-web-key-never-log"
    monkeypatch.setenv("TAVILY_API_KEY", secret)

    def respond(request):
        body = json.loads(request.content)
        if request.url.host == "api.tavily.com":
            assert request.headers["authorization"] == "Bearer " + secret
            web_requests.append((request.url.path, body))
            if request.url.path == "/search":
                if len(web_requests) == 1:
                    assert body["query"] == "Aurora project current status"
                    results = [
                        {"url": "https://irrelevant.example/", "title": "Distractor"},
                        {"url": first, "title": "Issuer project update"},
                    ]
                else:
                    assert body["query"] == "Aurora decision 84 final completion record"
                    assert body["include_domains"] == ["records.example"]
                    assert passages and "decision-84" in passages[0]["text"]
                    results = [{"url": linked, "title": "Decision 84 record"}]
                assert body["include_raw_content"] is False and body["include_answer"] is False
                return httpx.Response(200, json={"results": results, "usage": {"credits": 1}})
            assert body["urls"] in [[first], [linked]] and "query" not in body
            text = (
                "Aurora is pending. Completion record: "
                "[Decision 84](https://records.example/decision-84)."
                if body["urls"] == [first]
                else "Decision 84: Aurora completed on 2026-10-30."
            )
            return httpx.Response(
                200,
                json={
                    "results": [{"url": body["urls"][0], "raw_content": text}],
                    "usage": {"credits": 0.2},
                },
            )
        assert request.url.host == "api.deepseek.com"
        model_requests.append(body)
        step = len(model_requests)
        if step == 1:
            assert "decision-84" not in json.dumps(body)
            name, args = (
                "search",
                {
                    "query": "Aurora project current status",
                    "purpose": "Find status originals",
                    "intent": "initial",
                },
            )
        elif step == 8:
            assert body["messages"][0]["content"].startswith("REPORT_REVISION:")
            assert not body.get("tools")
            draft = json.loads(body["messages"][-1]["content"])["draft"]
        else:
            packets = [
                json.loads(m["content"].split("\n\n", 1)[1])
                for m in body["messages"]
                if isinstance(m.get("content"), str)
                and m["content"].startswith("Here is a summary")
            ]
            packet = packets[-1] if packets else None
            previous = (
                json.loads(
                    next(m for m in reversed(body["messages"]) if m["role"] == "tool")["content"]
                )
                if not packet
                else {}
            )
            if step == 2:
                name, args = "read_document", {"version_id": previous["results"][1]["version_id"]}
            elif step in {3, 6}:
                assert body.get("tool_choice") != {
                    "type": "function",
                    "function": {"name": "record_claim"},
                }  # voluntary recordings must not consume forced research turns
                doc = packet["documents"][0 if step == 3 else -1]
                pid, text = next(iter(doc["excerpts"].items()))
                passages.append({"passage_id": pid, "text": text})
                name, args = (
                    "record_claim",
                    {
                        "text": "Aurora remains pending."
                        if step == 3
                        else "Aurora completed on 2026-10-30.",
                        "passage_ids": [pid],
                        "event_time": None if step == 3 else "2026-10-30",
                        "stage": None if step == 3 else "completed",
                    },
                )
            elif step == 4:
                assert "decision-84" in passages[0]["text"]
                assert packet["source_link_leads"][0]["url"] == linked
                name, args = (
                    "search",
                    {
                        "query": "Aurora decision 84 final completion record",
                        "include_domains": ["records.example"],
                        "purpose": "Find dated original",
                        "intent": "gap",
                    },
                )
            elif step == 5:
                name, args = "open_url", {"url": packet["unread_candidates"][0]["url"]}
            else:
                assert step == 7 and len(packet["claims"]) == 2
                assert all(c["verification"] == "unverified" for c in packet["claims"])
        if step < 7:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"live-test-{step}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        else:
            message = {
                "role": "assistant",
                "content": draft
                if step == 8
                else f"Aurora completed on 2026-10-30. [{passages[-1]['passage_id']}]",
            }
        return httpx.Response(
            200,
            json={
                "id": f"live-test-{step}",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if step < 7 else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            },
        )

    original_init = harness.init_chat_model
    original_client = httpx.AsyncClient

    async def scenario():
        transport = httpx.MockTransport(respond)
        with httpx.Client(transport=transport) as sync_client:
            async with original_client(transport=transport) as async_client:
                monkeypatch.setattr(
                    harness,
                    "init_chat_model",
                    lambda *args, **kwargs: original_init(
                        *args,
                        api_key="test-model-key-never-log",
                        http_client=sync_client,
                        http_async_client=async_client,
                        **kwargs,
                    ),
                )
                monkeypatch.setattr(
                    retrieval,
                    "httpx",
                    SimpleNamespace(
                        AsyncClient=lambda **kw: original_client(transport=transport, **kw)
                    ),
                )
                return await harness.run_research(
                    tmp_path,
                    question_task("项目 Aurora 完成了吗？"),
                    budget_config=BudgetConfig(),
                    adapter="tavily",
                    model_name="deepseek:deepseek-flash",
                )

    result = asyncio.run(scenario())
    assert result["status"] == "complete" and result["model_finalized"], result
    assert result["budget"]["model_calls"] == len(model_requests) == 8
    assert result["budget"]["tokens"] == 8 * 130
    assert result["budget"]["retrieval_api_calls"] == len(web_requests) == 4
    assert result["budget"]["web_search_requests"] == 2
    assert result["budget"]["web_extract_requests"] == 2
    assert result["budget"]["retrieval_credits"] == pytest.approx(2.4)
    assert result["budget"].get("retrieval_unknown_usage_calls", 0) == 0
    assert not (tmp_path / "input_corpus.json").exists()
    evidence = read_json(tmp_path / "evidence.json")
    assert {d["url"] for d in evidence["documents"].values()} == {first, linked}
    activity = read_json(tmp_path / "live_activity.json")["events"]
    actual = [
        ("/" + e["operation"], e["arguments"])
        for e in activity
        if e["kind"] == "retrieval_dispatch"
    ]
    assert actual == web_requests
    tool_args = [
        e["arguments"] for e in activity if e["kind"] == "tool_dispatch" and e["name"] == "search"
    ]
    assert [a["query"] for a in tool_args] == [
        a["query"] for path, a in web_requests if path == "/search"
    ]
    assert "No prepared corpus" in (tmp_path / "activity.md").read_text()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
            assert b"test-model-key-never-log" not in path.read_bytes()

    # A completed run is read back; no paid model or web request is repeated.
    before = len(model_requests), len(web_requests)
    assert asyncio.run(harness.run_research(tmp_path, resume=True)) == result
    assert before == (len(model_requests), len(web_requests))
