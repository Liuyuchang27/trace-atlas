"""Actual DeepSeek/Tavily adapters under HTTP mocks; never external network calls."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from test_answer_contract import observation, task_at

import news_agent.harness as harness
import news_agent.retrieval as retrieval
from news_agent.schemas import BudgetConfig
from news_agent.storage import read_json


async def http_scenario(run, monkeypatch, mode):
    model_bodies, web_bodies = [], []
    first, closing = "https://regulator.example/consent", "https://issuer.example/closing"
    approval = "On October 13, 2023, the regulator gave Aurora consent to acquire Birch."
    completed = "On October 13, 2023, Aurora completed its acquisition of Birch."
    state = {"initial_vid": None, "bad_subject_sent": False}
    monkeypatch.setenv("TAVILY_API_KEY", "c3-test-dummy-web-key")

    def message_text(message):
        value = message.get("content") or ""
        return value if isinstance(value, str) else "".join(p.get("text", "") for p in value)

    def respond(request):
        body = json.loads(request.content)
        if request.url.host == "api.tavily.com":
            web_bodies.append((request.url.path, body))
            if request.url.path == "/search":
                assert body["query"] in {
                    "Aurora initial regulator record",
                    "Aurora Birch actual closing record",
                }
                url = first if body["query"] == "Aurora initial regulator record" else closing
                return httpx.Response(
                    200,
                    json={
                        "results": [{"url": url, "title": "Aurora record"}],
                        "usage": {"credits": 1},
                    },
                )
            assert request.url.path == "/extract"
            if mode == "source_failure" and body["urls"] == [closing]:
                return httpx.Response(503, json={"error": "fixture original unavailable"})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": body["urls"][0],
                            "raw_content": approval if body["urls"] == [first] else completed,
                        }
                    ],
                    "usage": {"credits": 0.2},
                },
            )
        assert request.url.host == "api.deepseek.com"
        model_bodies.append(body)
        actions, text = [], ""
        stage_message = message_text(body["messages"][-1])
        is_stage = stage_message.startswith("C3 bounded evidence step.")
        if message_text(body["messages"][0]).startswith("REPORT_REVISION:"):
            payload = json.loads(stage_message)
            docs = payload["eligible_evidence"]["documents"]
            doc = next((d for d in docs if d["url"] == closing), docs[0])
            pid, quote = next(iter(doc["excerpts"].items()))
            close = doc["url"] == closing
            text = json.dumps(
                {
                    "observations": [
                        observation(
                            quote,
                            passage_id=pid,
                            subject="Aurora",
                            kind="completion" if close else "approval",
                            target_event=close,
                        )
                    ]
                }
            )
        elif is_stage:
            packet = json.loads(stage_message.split("\n", 1)[1])
            stage = packet["repair_stage"]
            name = body["tool_choice"]["function"]["name"]
            assert [t["function"]["name"] for t in body["tools"]] == [name]
            if stage == "search":
                if mode == "model_timeout":
                    raise httpx.ReadTimeout("fixture unknown model usage", request=request)
                actions = [
                    (
                        "search",
                        {
                            "query": "Aurora Birch actual closing record",
                            "purpose": "Find dated actual closing original",
                            "intent": "gap",
                            "step_id": "main",
                        },
                    )
                ]
            elif stage == "read":
                actions = [("read_document", {"version_id": packet["candidates"][0]["version_id"]})]
            else:
                doc = next(
                    (d for d in packet["documents"] if d["url"] == closing), packet["documents"][0]
                )
                pid, quote = next(iter(doc["excerpts"].items()))
                close = doc["url"] == closing
                subject = "Aurora"
                if mode == "subject_retry" and close and not state["bad_subject_sent"]:
                    subject = "极光公司"
                    state["bad_subject_sent"] = True
                elif mode == "subject_retry" and close:
                    assert packet["last_tool_result"]["error"] == "subject_not_in_quote"
                    assert "original language" in packet["last_tool_result"]["recovery"]
                actions = [
                    (
                        "update_question_evidence",
                        {
                            "question_id": "main",
                            "seen_revision": packet["question_state"][0]["evidence_revision"],
                            "observation": observation(
                                quote,
                                passage_id=pid,
                                subject=subject,
                                kind="completion" if close else "approval",
                                target_event=close,
                            ),
                        },
                    )
                ]
        elif len(model_bodies) == 1:
            actions = [
                (
                    "search",
                    {
                        "query": "Aurora initial regulator record",
                        "purpose": "Find original evidence",
                        "intent": "initial",
                        "step_id": "main",
                    },
                )
            ]
        elif len(model_bodies) == 2:
            tool = next(m for m in reversed(body["messages"]) if m["role"] == "tool")
            state["initial_vid"] = json.loads(tool["content"])["results"][0]["version_id"]
            actions = [("read_document", {"version_id": state["initial_vid"]})]
        else:
            text = "Report from collected originals; missing completion evidence stays unknown."
        calls = [
            {
                "id": f"http-{len(model_bodies)}-{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(actions)
        ]
        message = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = calls
        return httpx.Response(
            200,
            json={
                "id": f"fixture-{len(model_bodies)}",
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
                "usage": {"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350},
            },
        )

    transport = httpx.MockTransport(respond)
    original_init, original_async = harness.init_chat_model, httpx.AsyncClient
    with httpx.Client(transport=transport) as sync_client:
        async with original_async(transport=transport) as async_client:
            model = original_init(
                "deepseek:deepseek-flash",
                api_key="c3-test-dummy-model-key",
                http_client=sync_client,
                http_async_client=async_client,
                temperature=0,
                max_tokens=1500,
                max_retries=0,
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
            monkeypatch.setattr(
                harness,
                "init_chat_model",
                lambda *args, **kwargs: model,
            )
            result = await harness.run_research(
                run,
                task_at().model_copy(update={"synthetic": True}),
                budget_config=BudgetConfig(),
                adapter="tavily",
                model_name="deepseek:deepseek-flash",
            )
    return result, model_bodies, web_bodies


@pytest.mark.parametrize("mode", ["success", "subject_retry", "source_failure", "model_timeout"])
def test_metered_same_agent_and_live_adapter_complete_bounded_repair_or_preserve_unknown(
    tmp_path, monkeypatch, mode
):
    from news_agent import model_retry

    async def skip_wait(_seconds):
        pass

    monkeypatch.setattr(model_retry, "wait_before_retry", skip_wait)
    result, models, web = asyncio.run(http_scenario(tmp_path, monkeypatch, mode))
    execution = read_json(tmp_path / "repair_execution.json")
    flow = execution["workflow"]
    assert flow["searches"] <= 1 and flow["reads"] <= 1 and flow["assessments"] <= 2
    assert flow["status"] == (
        "covered_provisionally" if mode in {"success", "subject_retry"} else "incomplete"
    ), result
    assert len(models) == result["budget"]["model_calls"]
    assert len(web) == result["budget"]["retrieval_api_calls"]
    assert sum(path == "/search" for path, _ in web) == (1 if mode == "model_timeout" else 2)
    assert result["budget"]["unknown_usage_calls"] == (
        1 + len(model_retry.RETRY_DELAYS) if mode == "model_timeout" else 0
    )
    assert result["budget"].get("retrieval_unknown_usage_calls", 0) == (
        1 if mode == "source_failure" else 0
    )
    assert not result["budget"]["pending"] and not result["budget"].get("retrieval_pending")
    answer = read_json(tmp_path / "answer_state.json")["inference"]["status"]
    assert answer == (
        "not_completed_by_cutoff" if mode in {"success", "subject_retry"} else "unknown"
    )
    assert result["budget"]["tokens"] <= 80000
    assert not (tmp_path / "input_corpus.json").exists()
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert b"c3-test-dummy-web-key" not in p.read_bytes()
            assert b"c3-test-dummy-model-key" not in p.read_bytes()
