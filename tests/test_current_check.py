"""Exercise recent first-party fact checks through the real SDK with mock providers."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

import news_agent.harness as harness
import news_agent.origin_pages as origin_pages
import news_agent.retrieval as retrieval
from news_agent.current_check import applies
from news_agent.current_report import report_draft, synthesize_notes
from news_agent.schemas import BudgetConfig, ResearchTask
from news_agent.storage import read_json
from news_agent.web_view import build_snapshot, report_html

OBSERVATION = (
    '\n\nRun observation context: {"local":"2026-11-20T09:00:00+08:00","timezone":"Asia/Hong_Kong"}'
)
URLS = {
    "latest": "https://publisher.example/launch",
    "fact": "https://publisher.example/ir/disclosure",
}
SNIPPETS = {
    "latest": "On November 19, 2026, Aurora opened worldwide on licensed platforms.",
    "fact": "On November 20, 2026, Aurora published its November operating notice.",
}
ORIGIN_INDEX = "https://publisher.example/"
ORIGIN_RELEASE = "https://publisher.example/news/aurora-now-available"
PREVIOUS_OFFICIAL = "https://publisher.example/ir/sales-update"
PREVIOUS_ESTIMATE = "https://news.example/aurora-preorders"


def task(question):
    return ResearchTask(
        task_id="current-http",
        event=question,
        question=question + OBSERVATION,
        mode="retrospective",
        synthetic=True,
    )


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Aurora 最新的销量", True),
        ("What is the current launch status for Aurora?", True),
        ("最新美国就业数据及近半年趋势", False),
        ("截至 2023 年的最新销量", False),
        ("What was released next?", False),
    ],
)
def test_current_intent_preserves_historical_and_trend_tasks(question, expected):
    assert bool(applies(task(question))) is expected


@pytest.mark.parametrize(
    "outcome,route",
    [
        ("confirmed", "search"),
        ("not_found", "search"),
        ("not_found", "retry"),
        ("not_found", "direct"),
        ("insufficient", "search"),
        ("not_found", "fresh"),
        ("insufficient", "stale"),
        ("confirmed", "previous"),
        ("not_found", "estimate"),
    ],
)
def test_current_answer_is_scoped_cited_and_closes_without_temporal_loop(
    tmp_path, monkeypatch, outcome, route
):
    second_route = route == "retry"
    refresh = route in {"fresh", "stale", "previous"}
    earlier_read = route in {"previous", "estimate"}
    prior_url = PREVIOUS_OFFICIAL if route == "previous" else PREVIOUS_ESTIMATE
    prior_id = "web-" + hashlib.sha256(prior_url.encode()).hexdigest()[:20]
    current_outcome = "insufficient" if route == "previous" else outcome
    web, models = [], []
    origin_reads = []
    versions = {
        key: "web-" + hashlib.sha256(url.encode()).hexdigest()[:20] for key, url in URLS.items()
    }
    origin_id = "origin-" + hashlib.sha256(ORIGIN_RELEASE.encode()).hexdigest()[:20]

    async def origin(url, domains, timeout):
        assert set(domains) == {"publisher.example"}
        origin_reads.append(url)
        if url == ORIGIN_INDEX:
            link = (
                '<a href="/news/aurora-now-available">Aurora Now Available</a>'
                if route == "fresh"
                else ""
            )
            return (
                url,
                f"<title>Aurora press</title><main>{link}<p>Recent updates.</p></main>".encode(),
            )
        assert url == ORIGIN_RELEASE
        article = """<title>Aurora Now Available</title><main>
            <h1>Aurora Now Available</h1><time>November 19, 2026</time>
            <p>Aurora is now available worldwide on licensed platforms.</p></main>"""
        return url, article.encode()

    monkeypatch.setattr(origin_pages, "fetch_origin_html", origin)
    originals = {
        **SNIPPETS,
        "fact": (
            "On November 20, 2026, Aurora disclosed worldwide retail sales of 246,000 units."
            if current_outcome == "confirmed"
            else SNIPPETS["fact"]
        ),
    }
    if refresh:
        originals["latest"] = (
            "# Pre-Load Aurora Now\n\nNovember 10, 2026\n\n"
            "Pre-load Aurora today before its planned November 19, 2026 release."
        )
    refused = False
    estimate_refused = False

    def respond(request):
        nonlocal refused, estimate_refused
        args = json.loads(request.content)
        if request.url.host == "api.tavily.com":
            web.append((request.url.path, args))
            if request.url.path == "/search":
                if args.get("include_domains_mode") == "prefer":
                    return httpx.Response(
                        200,
                        json={
                            "results": [
                                {
                                    "url": prior_url,
                                    "title": "Aurora previous report",
                                    "published_date": "Thu, 19 Nov 2026 00:00:00 GMT",
                                }
                            ]
                            if earlier_read
                            else [],
                            "usage": {"credits": 1},
                        },
                    )
                key = "latest" if "launch" in args["query"] else "fact"
                hits = [
                    {
                        "url": URLS[key],
                        "title": key,
                        "published_date": "Fri, 20 Nov 2026 00:00:00 GMT",
                    },
                    {
                        "url": "https://publisher.example/future",
                        "title": "future",
                        "published_date": "Sat, 21 Nov 2026 00:00:00 GMT",
                    },
                ]
                return httpx.Response(
                    200,
                    json={
                        "results": [] if route in {"retry", "direct"} and len(web) <= 2 else hits,
                        "usage": {"credits": 1},
                    },
                )
            if args["urls"] == [prior_url]:
                text = (
                    "Published November 19, 2026. As of November 17, 2026, "
                    "Aurora sold 243,000 units at retail. Aurora recorded 153,000 pre-orders."
                    if route == "previous"
                    else (
                        "An analyst estimates Aurora pre-orders reached 12,000 units, "
                        "with net revenue of $2.4 million."
                    )
                )
                return httpx.Response(
                    200,
                    json={
                        "results": [{"url": prior_url, "raw_content": text}],
                        "usage": {"credits": 1},
                    },
                )
            key = "latest" if args["urls"] == [URLS["latest"]] else "fact"
            return httpx.Response(
                200,
                json={
                    "results": [{"url": URLS[key], "title": key, "raw_content": originals[key]}],
                    "usage": {"credits": 1},
                },
            )
        available = {t["function"]["name"] for t in args.get("tools", [])}
        models.append(available)
        name, payload = None, None
        if len(models) <= (2 if second_route else 1):
            name = "search_current_sources"
            payload = {
                "latest_update_query": "Aurora launch notice",
                "requested_fact_query": "Aurora current worldwide retail sales units disclosure",
                "authority_domains": ["publisher.example"],
            }
            if len(models) == 2:
                assert available == {name, "read_current_sources"}
                payload["latest_update_query"] = "Aurora launch announcement investor relations"
                payload["requested_fact_query"] = "Aurora operating results official units sold"
        elif len(models) == (3 if second_route else 2):
            name = "read_current_sources"
            payload = {"version_ids": list(versions.values())}
            if route == "direct":
                payload = {
                    "official_pages": [
                        {"url": URLS[key], "focus": "Aurora", "role": role}
                        for key, role in [("latest", "latest_update"), ("fact", "requested_fact")]
                    ]
                }
        elif refresh and len(models) == 3:
            name = "refresh_current_status"
            assert available == {name}
            payload = {
                "official_pages": [
                    {"url": ORIGIN_INDEX, "focus": "Aurora release", "role": "latest_update"}
                ]
            }
        elif route == "fresh" and len(models) == 4:
            name = "read_current_sources"
            assert available == {name}
            payload = {"version_ids": [origin_id]}
        elif available == {"search_previous_sales"}:
            name = "search_previous_sales"
            payload = {"query": "Aurora actual units sold in the latest reporting period"}
        elif "record_current_result" in available:
            name = "record_current_result"
            passages = read_json(tmp_path / "evidence.json")["passages"]
            refs = list(passages)
            if refresh:
                delivered = read_json(tmp_path / "budget.json")["delivered_passages"]
                refs = list({passages[pid]["version_id"]: pid for pid in delivered}.values())
            payload = {
                "outcome": current_outcome,
                "text": (
                    "截至所查原始披露，全球零售销量为 246,000 份。"
                    if current_outcome == "confirmed"
                    else "已查阅发行方上线公告和运营公告，未核实所问销量。"
                ),
                "passage_ids": refs,
                "record_on": "2026-11-20" if current_outcome == "confirmed" else None,
            }
            previous = read_json(tmp_path / "adaptive_investigation.json")["current_check"].get(
                "previous_period", {}
            )
            if previous.get("stage") == "assess":
                old = [pid for pid, row in passages.items() if row["version_id"] == prior_id]
                if route == "previous":
                    payload = {
                        "outcome": "confirmed",
                        "text": "上一次核实的全球零售销量为 243,000 份。",
                        "record_on": "2026-11-19",
                        "data_through": "2026-11-17",
                        "passage_ids": old,
                    }
                else:
                    payload["passage_ids"] = [*refs, *old]
                    if not estimate_refused:
                        payload.update(
                            outcome="confirmed",
                            record_on="2026-11-20",
                            text="实际销量为 12,000 份。",
                        )
                        estimate_refused = True
            if len(models) == 3 and outcome == "not_found" and route == "search":
                payload["passage_ids"] = refs[:1]
                refused = True
            if route == "stale":
                payload["text"] = "Aurora 尚未发售，尚无实际销量。"
        message = {"role": "assistant", "content": "本轮核查完成。"}
        if args["messages"][0]["content"].startswith("CURRENT_REPORT_SYNTHESIS"):
            assert not available and args["response_format"] == {"type": "json_object"}
            notes = json.loads(args["messages"][-1]["content"])
            assert len(notes) == 1 and "{1}" in notes[0]["original"]
            summary = (
                "公司披露预购数量为 {1} 份。"
                if route == "previous"
                else "分析师估算，预购量约为 {1} 份，预购净收入约为 {2}。"
            )
            message["content"] = json.dumps(
                {"preorders": [{"passage_id": notes[0]["passage_id"], "summary": summary}]},
                ensure_ascii=False,
            )
        if name:
            assert name in available, available
            message["content"] = ""
            message["tool_calls"] = [
                {
                    "id": f"call-{len(models)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(payload)},
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": f"answer-{len(models)}",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if name else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350},
            },
        )

    async def exercise():
        transport = httpx.MockTransport(respond)
        client_class = httpx.AsyncClient
        init = harness.init_chat_model
        async with client_class(transport=transport) as client:
            model = init(
                "deepseek:deepseek-flash",
                api_key="offline-key",
                http_async_client=client,
                max_retries=0,
            )
            monkeypatch.setenv("TAVILY_API_KEY", "offline-tavily-key")
            monkeypatch.setattr(harness, "init_chat_model", lambda *a, **kw: model)
            monkeypatch.setattr(
                retrieval,
                "httpx",
                SimpleNamespace(AsyncClient=lambda **kw: client_class(transport=transport, **kw)),
            )
            return await harness.run_research(
                tmp_path,
                task("Aurora 最新的销量"),
                None,
                BudgetConfig(),
                adapter="tavily",
                model_name="deepseek:deepseek-flash",
                workflow="adaptive-event-map",
            )

    result = asyncio.run(exercise())
    assert result["status"] == ("partial" if outcome == "insufficient" else "complete"), result
    assert result["stop_reason"] == (
        "current_check_incomplete" if outcome == "insufficient" else "current_fact_checked"
    )
    assert result["report_source"] == "current_fact_check"
    historical_requests = [] if current_outcome == "confirmed" else ["/search"]
    if earlier_read:
        historical_requests.append("/extract")
    assert [path for path, _ in web] == ["/search"] * (4 if second_route else 2) + [
        "/extract"
    ] * 2 + historical_requests
    assert web[0][1]["start_date"] == "2026-11-12"
    assert web[0][1]["topic"] == "news"
    assert web[1][1]["start_date"] == "2026-10-20"
    assert web[0][1]["end_date"] == "2026-11-21"
    assert web[1][1]["include_domains"] == ["publisher.example"]
    assert web[0][1]["filter_by_published_date"] is True
    assert not (tmp_path / "temporal_review.json").exists()
    expected_calls = {"fresh": 6, "stale": 5, "previous": 5}.get(
        route, 5 if refused or second_route else 4
    )
    expected_calls += bool(historical_requests) + earlier_read + estimate_refused
    assert len(models) == expected_calls
    view = build_snapshot(tmp_path)
    assert view["report_brief"]["current"]["outcome"] == outcome
    attempts = view["adaptive"]["current_check"]["attempts"]
    assert all(set(ids) <= set(versions.values()) for a in attempts for ids in a["hits"].values())
    assert len(view["report_brief"]["sources"]) == 2 + len(origin_reads) + earlier_read
    assert len(view["report_brief"]["searches"]) == (2 if second_route else 1)
    html = report_html(view)
    markdown = (tmp_path / "report.md").read_text()
    assert "核查截至 2026-11-20 09:00（香港时间）" in html
    assert URLS["latest"] in html and URLS["fact"] in html
    assert "原始记录日期" in html if outcome == "confirmed" else "原始记录日期" not in html
    if route == "previous":
        assert "最近一次可核实的实际数据" in html and "243,000" in html
        assert "数据截至 2026-11-17" in html and "原始记录日期 2026-11-19" in html
        assert PREVIOUS_OFFICIAL in html
        for report in (html, markdown):
            assert report.index("243,000") < report.index("153,000")
    if route == "estimate":
        assert "第三方预购估算" in html and "12,000" in html
        assert "最近一次可核实的实际数据" not in html
        for report in (html, markdown):
            assert report.index("12,000") < report.index("核查结论") < report.index("核查截至")
            assert "分析师估算，预购量约为 12,000份，预购净收入约为 $240 万。" in report
        note = view["report_brief"]["preorder_notes"][0]
        assert f'href="#{note["passage_id"]}"' in html.split("核查结论", 1)[0]
        assert (tmp_path / "current_report_draft.json").is_file()
        assert "An analyst" not in view["report_brief_html"]
    if refresh:
        assert origin_reads == (
            [ORIGIN_INDEX, ORIGIN_RELEASE] if route == "fresh" else [ORIGIN_INDEX]
        )
        assert "尚未发售" not in view["report_brief"]["current"]["text"]
        if route == "fresh":
            release = view["report_brief"]["current"]["release"]
            assert release["day"] == "2026-11-19" and release["url"] == ORIGIN_RELEASE
            assert release["references"]
            assert "已发售" in html
            assert html.index("公告确认已发售") < html.index("核查结论")
        elif route == "stale":
            assert "发售状态待确认" in html
            assert "release" not in view["report_brief"]["current"]


def test_recent_report_cannot_replace_or_swap_source_numbers():
    notes = [{"passage_id": "source", "text": "About 2.5 million pre-orders and $3 billion net."}]
    draft = {
        "preorders": [{"passage_id": "source", "summary": "预购量约为 {1} 份，净收入为 {2}。"}]
    }
    bound = report_draft(notes, json.dumps(draft))
    assert synthesize_notes(notes, bound)[0]["summary"] == "预购量约为 250 万份，净收入为 $30 亿。"
    assert (
        synthesize_notes([{**notes[0], "text": "1 million pre-orders."}], bound)[0].get("summary")
        is None
    )
    for unsafe in ("预购量约为 {2}，净收入为 {1}。", "预购量约为 {1}，净收入为 400 万。"):
        draft["preorders"][0]["summary"] = unsafe
        with pytest.raises(ValueError, match="preorder_summary_not_bound"):
            report_draft(notes, json.dumps(draft))
