"""Synthetic regressions for bounded contexts and truncated staged JSON replies."""

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage
from test_current_check import task as current_task
from test_event_graph import original, setup

import news_agent.cli as cli
import news_agent.harness as harness
import news_agent.model_routing as routing
import news_agent.retrieval as retrieval
from news_agent.adaptive import AdaptiveInvestigation
from news_agent.context import byte_size
from news_agent.core_context import CoreEvidenceContext
from news_agent.event_graph import Finding, QuestionPart
from news_agent.mock_model import FixtureChatModel
from news_agent.storage import read_json


@pytest.mark.parametrize("writing", [False, True])
def test_large_multilingual_plan_compacts_without_losing_originals(tmp_path, writing):
    task, evidence, budget, graph = setup(tmp_path)
    evidence.event_graph = graph
    evidence.adaptive = AdaptiveInvestigation(graph, budget)
    parts = [
        QuestionPart(
            part_id=f"part-{i}",
            question=f"{i}: " + "核实公开原始记录的时间、范围与实际结果。" * 8,
            evidence_needed="查阅原始记录并保留限定条件。" * 12,
        )
        for i in range(6)
    ]
    graph.record(
        "question",
        "Synthetic research question",
        "Aurora",
        "question",
        "major",
        "Answer all requested parts",
        Finding(
            assertion="Open questions",
            stance="unknown",
            occurrence="unknown",
            uncertainty="No original evidence yet",
        ),
        0,
        parts,
    )
    evidence.adaptive.refresh()
    ref = original(evidence, budget, "source", "Aurora proposed a project, subject to approval.")
    # Saved metadata can also grow independently of the visible plan.
    for i in range(80):
        original(evidence, budget, f"saved-{i}", "Unread synthetic record.", delivered=False)
    if writing:
        budget.finish_research("fixture_complete")
    before = copy.deepcopy(graph.data)
    context = CoreEvidenceContext(FixtureChatModel(), task, evidence, budget)
    # The SDK middleware hook, not just a hand-built serializer call.
    result = asyncio.run(context.abefore_model({"messages": [HumanMessage(task.question)]}, None))
    assert result is not None
    active = budget.state["active_context_packet"]
    packet = read_json(tmp_path / active["path"])
    assert byte_size(packet) <= context.limit == 6000
    assert packet["task"]["question"] == task.question
    visible = {pid: text for d in packet["documents"] for pid, text in d["excerpts"].items()}
    assert visible == {ref.passage_id: ref.quote}
    assert packet["inventory"]["context_metadata_omitted"]
    assert packet["inventory"]["saved_documents"] == 81
    assert graph.data == before  # compaction must not rewrite durable research state
    assert budget.state["model_calls"] == 0


@pytest.mark.parametrize("review_cap", [None, 16000])
def test_cli_current_review_truncation_is_metered_and_stage_cap_is_configurable(
    tmp_path, monkeypatch, capsys, review_cap
):
    urls = ["https://publisher.example/launch", "https://publisher.example/sales"]
    versions = ["web-" + hashlib.sha256(url.encode()).hexdigest()[:20] for url in urls]
    runs = tmp_path / "runs"
    requests = []
    originals = [
        "On November 19, 2026, Aurora opened worldwide on licensed platforms.",
        "On November 20, 2026, Aurora disclosed worldwide retail sales of 246,000 units.",
    ]

    def respond(request):
        body = json.loads(request.content)
        if request.url.host == "api.tavily.com":
            if request.url.path == "/search":
                index = 0 if "launch" in body["query"] else 1
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "url": urls[index],
                                "title": "Synthetic disclosure",
                                "published_date": "Fri, 20 Nov 2026 00:00:00 GMT",
                            }
                        ],
                        "usage": {"credits": 1},
                    },
                )
            assert request.url.path == "/extract"
            index = urls.index(body["urls"][0])
            return httpx.Response(
                200,
                json={
                    "results": [{"url": urls[index], "raw_content": originals[index]}],
                    "usage": {"credits": 1},
                },
            )
        assert request.url.host == "api.deepseek.com"
        requests.append(body)
        name = None
        if len(requests) == 1:
            name = "search_current_sources"
            args = {
                "latest_update_query": "Aurora launch notice",
                "requested_fact_query": "Aurora worldwide retail sales disclosure",
                "authority_domains": ["publisher.example"],
            }
        elif len(requests) == 2:
            name, args = "read_current_sources", {"version_ids": versions}
        elif len(requests) == 3:
            run = next(runs.iterdir())
            passages = read_json(run / "evidence.json")["passages"]
            name, args = (
                "record_current_result",
                {
                    "outcome": "confirmed",
                    "text": "Aurora is available worldwide on licensed platforms.",
                    "passage_ids": list(passages),
                    "record_on": "2026-11-20",
                },
            )
        else:
            assert len(requests) == 4
            assert body["thinking"] == {"type": "enabled"}
            assert body["reasoning_effort"] == "high"
            assert body["response_format"] == {"type": "json_object"}
            assert not body.get("tools")
        message = {"role": "assistant", "content": ""}
        if name:
            assert name in {t["function"]["name"] for t in body["tools"]}
            message["tool_calls"] = [
                {
                    "id": f"call-{len(requests)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
            finish = "tool_calls"
        else:
            payload = json.loads(body["messages"][-1]["content"])
            message["content"] = (
                json.dumps(
                    {
                        "answer": "Aurora is available worldwide on licensed platforms.",
                        "answer_passages": [r["id"] for r in payload["eligible_originals"]],
                        "preorders": [],
                    }
                )
                if body["max_tokens"] >= 8000
                else '{"answer":'
            )
            message["reasoning_content"] = "PRIVATE_REASONING_CANARY"
            finish = "stop" if body["max_tokens"] >= 8000 else "length"
        return httpx.Response(
            200,
            json={
                "id": f"offline-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350},
            },
        )

    transport = httpx.MockTransport(respond)
    original_async = httpx.AsyncClient
    init = harness.init_chat_model
    with httpx.Client(transport=transport) as sync_client:
        async_client = original_async(transport=transport)

        def make_model(name, **kwargs):
            return init(
                name,
                api_key="offline-placeholder",
                http_client=sync_client,
                http_async_client=async_client,
                **kwargs,
            )

        monkeypatch.setattr(harness, "init_chat_model", make_model)
        monkeypatch.setattr(routing, "init_chat_model", make_model)
        monkeypatch.setattr(
            retrieval,
            "httpx",
            SimpleNamespace(AsyncClient=lambda **kw: original_async(transport=transport, **kw)),
        )
        monkeypatch.setattr(cli, "load_dotenv", lambda **kw: None)
        # Freeze observation time, so source freshness does not depend on test date.
        monkeypatch.setattr(cli, "question_task", lambda question, *a: current_task(question))
        monkeypatch.setenv("TAVILY_API_KEY", "offline-placeholder")
        args = [
            "ask",
            "What is the current launch status for Aurora?",
            "--workflow",
            "adaptive-event-map",
            "--model-profile",
            routing.PROFILE,
            "--model",
            "deepseek:deepseek-flash",
            "--runs-dir",
            str(runs),
        ]
        if review_cap:
            config = tmp_path / "budget.json"
            config.write_text(json.dumps({"review_output_tokens": review_cap}))
            args += ["--budget", str(config)]
        try:
            exit_code = cli.main(args)
        finally:
            asyncio.run(async_client.aclose())
    displayed = json.loads(capsys.readouterr().out)
    assert exit_code == 0, displayed
    assert displayed["status"] == ("complete" if review_cap else "partial")
    assert displayed["stop_reason"] == ("current_fact_checked" if review_cap else "output_limit")
    assert displayed["model_finalized"] is bool(review_cap)
    assert [r["max_tokens"] for r in requests] == [1500, 1500, 1500, review_cap or 1500]
    run = next(runs.iterdir())
    budget = read_json(run / "budget.json")
    assert budget["model_calls"] == 4 and budget["tokens"] == 1400
    assert budget["unknown_usage_calls"] == 0 and not budget["pending"]
    assert "worldwide" in (run / "report.md").read_text()
    for path in run.rglob("*"):
        if path.suffix in {".json", ".jsonl", ".md", ".txt"}:
            assert "PRIVATE_REASONING_CANARY" not in path.read_text()


def test_cli_truncated_planning_stops_once_with_known_usage(tmp_path, monkeypatch, capsys):
    requests = []

    def respond(request):
        assert request.url.host == "api.deepseek.com"
        body = json.loads(request.content)
        requests.append(body)
        assert body["response_format"] == {"type": "json_object"}
        assert body["thinking"] == {"type": "enabled"}
        assert not body.get("tools")
        return httpx.Response(
            200,
            json={
                "id": "offline-planning",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": '{"scope":',
                            "reasoning_content": "PRIVATE_REASONING_CANARY",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350},
            },
        )

    transport = httpx.MockTransport(respond)
    init = harness.init_chat_model
    async_client = httpx.AsyncClient(transport=transport)
    with httpx.Client(transport=transport) as sync_client:

        def make_model(name, **kwargs):
            return init(
                name,
                api_key="offline-placeholder",
                http_client=sync_client,
                http_async_client=async_client,
                **kwargs,
            )

        monkeypatch.setattr(harness, "init_chat_model", make_model)
        monkeypatch.setattr(routing, "init_chat_model", make_model)
        monkeypatch.setattr(cli, "load_dotenv", lambda **kw: None)
        monkeypatch.setenv("TAVILY_API_KEY", "offline-placeholder")
        try:
            code = cli.main(
                [
                    "ask",
                    "How did the Aurora project develop?",
                    "--workflow",
                    "adaptive-event-map",
                    "--model-profile",
                    routing.PROFILE,
                    "--model",
                    "deepseek:deepseek-flash",
                    "--runs-dir",
                    str(tmp_path / "runs"),
                ]
            )
        finally:
            asyncio.run(async_client.aclose())
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["status"] == "partial"
    assert result["stop_reason"] == "output_limit" and not result["model_finalized"]
    assert len(requests) == result["usage"]["model_calls"] == 1
    assert result["usage"]["tokens"] == 350
    run = next((tmp_path / "runs").iterdir())
    budget = read_json(run / "budget.json")
    assert budget["unknown_usage_calls"] == 0 and not budget["pending"]
    assert not read_json(run / "event_graph.json")["nodes"]
    assert "PRIVATE_REASONING_CANARY" not in (run / "trajectory.jsonl").read_text()


def test_adaptive_compaction_still_refuses_an_intrinsically_oversized_task(tmp_path):
    from news_agent.budget import BudgetExceeded

    task, evidence, budget, graph = setup(tmp_path)
    evidence.event_graph = graph
    evidence.adaptive = AdaptiveInvestigation(graph, budget)
    task = task.model_copy(update={"question": "Long synthetic question. " * 1000})
    context = CoreEvidenceContext(FixtureChatModel(), task, evidence, budget)
    with pytest.raises(BudgetExceeded, match="evidence_context_index_limit"):
        context.packet([])
