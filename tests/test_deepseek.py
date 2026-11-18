import asyncio
import json

import httpx
import pytest

import news_agent.harness as harness
from news_agent.schemas import BudgetConfig
from news_agent.storage import read_json


@pytest.mark.parametrize("model_id", ["deepseek-flash", "deepseek-v4-pro"])
@pytest.mark.parametrize("stop_after_read", [False, True])
@pytest.mark.parametrize("final_reason", ["stop", "length"])
def test_deepseek_tool_cycle_and_usage_without_network(
    tmp_path, task, corpus_path, monkeypatch, stop_after_read, final_reason, model_id
):
    requests = []
    passage = {}

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.host == "api.deepseek.com"
        assert request.headers["authorization"] == "Bearer test-not-a-real-key"
        assert body["model"] == model_id
        assert body["thinking"] == {"type": "disabled"}
        assert body.get("max_tokens", body.get("max_completion_tokens")) == 1500
        if len(requests) == 5:
            assert not body.get("tools")
            assert body["messages"][0]["content"].startswith("REPORT_REVISION:")
            draft = json.loads(body["messages"][-1]["content"])["draft"]
            return httpx.Response(
                200,
                json={
                    "id": "revision",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": draft},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
                },
            )
        tool_names = {t["function"]["name"] for t in body["tools"]}
        assert tool_names <= {
            "search",
            "read_document",
            "record_claim",
            "get_passage",
            "list_evidence",
            "write_todos",
        }
        step = len(requests)
        if stop_after_read and step >= 3:
            assert tool_names == {"record_claim", "get_passage", "list_evidence"}
        if step == 1:
            name, arguments = "search", {"query": "Aurora Birch"}
        else:
            latest_tool = next((m for m in reversed(body["messages"]) if m["role"] == "tool"), None)
            if latest_tool:
                previous = json.loads(latest_tool["content"])
            else:
                summary = next(
                    m["content"]
                    for m in body["messages"]
                    if isinstance(m.get("content"), str)
                    and m["content"].startswith("Here is a summary")
                )
                packet = json.loads(summary.split("\n\n", 1)[1])
                if step == 3:
                    row = packet["documents"][0]
                    pid, text = next(iter(row["excerpts"].items()))
                    previous = {"passages": [{"passage_id": pid, "text": text}]}
                else:
                    previous = {"claim": packet["claims"][0]}
            if step == 2:
                name, arguments = (
                    "read_document",
                    {"version_id": previous["results"][0]["version_id"]},
                )
            elif step == 3:
                passage.update(previous["passages"][0])
                name, arguments = (
                    "record_claim",
                    {"text": passage["text"], "passage_ids": [passage["passage_id"]]},
                )
            else:
                assert step == 4
                assert previous["claim"]["verification"] == "unverified"
        message = {"role": "assistant", "content": ""}
        if step < 4:
            message["tool_calls"] = [
                {
                    "id": f"deepseek-test-{step}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        else:
            message["content"] = f"HTTP-MOCK fixture: {passage['text']} [{passage['passage_id']}]"
        return httpx.Response(
            200,
            json={
                "id": f"test-completion-{step}",
                "object": "chat.completion",
                "created": 0,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if step < 4 else final_reason,
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            },
        )

    original_init = harness.init_chat_model
    original_wrap_tool = harness.ResearchBoundary.awrap_tool_call

    async def wrap_tool(boundary, request, handler):
        result = await original_wrap_tool(boundary, request, handler)
        if stop_after_read and request.tool_call["name"] == "read_document":
            boundary.budget.finish_research("retrieval_no_new_versions")
        return result

    monkeypatch.setattr(harness.ResearchBoundary, "awrap_tool_call", wrap_tool)

    async def scenario():
        transport = httpx.MockTransport(respond)
        with httpx.Client(transport=transport) as sync_client:
            async with httpx.AsyncClient(transport=transport) as async_client:
                monkeypatch.setattr(
                    harness,
                    "init_chat_model",
                    lambda *args, **kwargs: original_init(
                        *args,
                        api_key="test-not-a-real-key",
                        http_client=sync_client,
                        http_async_client=async_client,
                        **kwargs,
                    ),
                )
                return await harness.run_research(
                    tmp_path,
                    task,
                    corpus_path,
                    BudgetConfig(),
                    model_name="deepseek:" + model_id,
                )

    result = asyncio.run(scenario())
    partial = stop_after_read or final_reason == "length"
    assert result["status"] == ("partial" if partial else "complete"), result
    assert result["model_finalized"] == (final_reason == "stop")
    assert result["report_source"] == "model"
    if final_reason == "length":
        assert result["stop_reason"] == "output_limit"
    elif stop_after_read:
        assert result["stop_reason"] == "retrieval_no_new_versions"
    expected_calls = 5 if final_reason == "stop" else 4
    assert len(requests) == result["budget"]["model_calls"] == expected_calls
    assert result["budget"]["tokens"] == 130 * expected_calls
    assert result["budget"]["unknown_usage_calls"] == 0
    assert result["api_cost_usd"] is None
    assert read_json(tmp_path / "claims.json")
    assert "HTTP-MOCK fixture" in (tmp_path / "report.md").read_text()
