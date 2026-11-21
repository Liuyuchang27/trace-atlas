import asyncio
import json
import re
from types import SimpleNamespace

import httpx
from langchain_core.messages import ToolMessage
from test_event_graph import original, setup

from news_agent import retrieval
from news_agent.adaptive import AdaptiveInvestigation
from news_agent.event_graph import Finding, TimeSpan, graph_tools
from news_agent.harness import ResearchBoundary, make_tools
from news_agent.progress import visual_search_tool
from news_agent.research_batch import pending_targets
from news_agent.retrieval import TavilyCorpus
from news_agent.web_view import build_snapshot, report_html


def initialize(tmp_path):
    task, evidence, budget, graph = setup(tmp_path)
    adaptive = AdaptiveInvestigation(graph, budget)
    adaptive.parallel_search_enabled = True
    evidence.adaptive = adaptive
    begin = next(t for t in graph_tools(graph) if t.name == "begin_investigation")
    return task, evidence, budget, graph, adaptive, begin


def plan(parts):
    return {
        "node_id": "question",
        "title": "Synthetic evidence investigation",
        "scope": {
            "direction": "both",
            "anchor": "latest",
            "objective": "answer",
            "minimum_event_anchors": 0,
        },
        "question_parts": parts,
    }


def part(part_id, **kwargs):
    return {
        "part_id": part_id,
        "question": f"Establish {part_id}",
        "evidence_needed": f"An original record covering {part_id}",
        **kwargs,
    }


def bind(graph, ref, part_id, *, month=None, value=None, complete=True):
    graph.append(
        "question",
        Finding(
            assertion="An original record gives the requested observation.",
            stance="supports",
            occurrence="observed",
            originals=[ref],
            question_part_id=part_id,
            answers_question_part=complete,
            event_time=TimeSpan(precision="month", start=month) if month else TimeSpan(),
            monthly_value=value,
        ),
        len(graph.data["nodes"]["question"]["findings"]),
    )


def test_scope_requires_evidence_and_keeps_dependent_work_out_of_first_batch(tmp_path):
    _, evidence, budget, graph, adaptive, begin = initialize(tmp_path)
    args = [part("status"), part("timeline"), part("driver", depends_on=["timeline"])]
    invalid = [*args[:2], part("driver", depends_on=["missing"])]
    assert "depends_on" in json.loads(asyncio.run(begin.ainvoke(plan(invalid))))["error"]
    assert (
        "evidence_needed"
        in json.loads(
            asyncio.run(
                begin.ainvoke(
                    plan(
                        [
                            {"part_id": "status", "question": "Missing evidence requirement"},
                        ]
                    )
                )
            )
        )["error"]
    )
    assert not graph.data["scope_history"]
    assert "error" not in json.loads(asyncio.run(begin.ainvoke(plan(args))))
    adaptive.refresh()
    assert [g["question_part_id"] for g in pending_targets(adaptive)] == ["status", "timeline"]
    assert adaptive.summary()["waiting_for_prerequisites"][0]["blocked_by"] == ["timeline"]
    assert [t.name for t in adaptive.surface(adaptive.tools())] == ["search_evidence_batch"]
    ref = original(evidence, budget, "record", "The publication describes the full timeline.")
    bind(graph, ref, "timeline")
    adaptive.refresh()
    assert [g["question_part_id"] for g in pending_targets(adaptive)] == ["driver", "status"]
    assert "waiting_for_prerequisites" not in adaptive.summary()


def test_shared_month_anchor_does_not_copy_one_value_into_distinct_metrics(tmp_path):
    _, evidence, budget, graph, adaptive, begin = initialize(tmp_path)
    series = {"periods": 6, "anchor_part_id": "anchor", "unit": "%"}
    args = [
        part("anchor"),
        part("annual", monthly_series=series),
        part("monthly", monthly_series=series),
    ]
    assert "error" not in json.loads(asyncio.run(begin.ainvoke(plan(args))))
    ref = original(evidence, budget, "rates", "In October 2026 annual change was 2.7 percent.")
    bind(graph, ref, "anchor", month="2026-10", value=2.7)
    bind(graph, ref, "annual", month="2026-10", value=2.7, complete=False)
    parts = {p["part_id"]: p for p in graph.node_view("question")["part_status"]}
    assert parts["annual"]["month_coverage"]["points"]["2026-10"]["value"] == 2.7
    assert "2026-10" in parts["monthly"]["month_coverage"]["missing"]
    adaptive.refresh()
    newest = [
        g["question_part_id"]
        for g in pending_targets(adaptive)
        if g.get("reference_month") == "2026-10"
    ]
    assert newest == ["monthly"]


def test_http_fans_out_monthly_requests_and_preserves_other_lines_on_failure(tmp_path, monkeypatch):
    task, evidence, budget, graph, adaptive, begin = initialize(tmp_path)
    args = [
        part("anchor"),
        part(
            "rates",
            depends_on=["anchor"],
            monthly_series={
                "periods": 12,
                "anchor_part_id": "anchor",
                "unit": "%",
            },
        ),
    ]
    assert "error" not in json.loads(asyncio.run(begin.ainvoke(plan(args))))
    bind(
        graph,
        original(evidence, budget, "anchor", "The latest reference period is October 2026."),
        "anchor",
        month="2026-10",
    )
    adaptive.refresh()
    targets = pending_targets(adaptive)
    assert len(targets) == 12
    corpus = TavilyCorpus(task, "mock-key", store=evidence.store, budget=budget)
    budget.state["evidence_chain_version"] = "test"
    budget.state["search_round_limit"] = None
    tools = make_tools(corpus, evidence, budget, stop_guard=lambda: False)
    adaptive.search = visual_search_tool(
        next(t for t in tools if t.name == "search"), evidence, budget
    )
    adaptive.reader = next(t for t in tools if t.name == "read_document")
    peak = {"/search": 0, "/extract": 0}

    async def exercise():
        in_flight = dict.fromkeys(peak, 0)
        gates = {kind: asyncio.Event() for kind in peak}

        async def respond(request):
            kind = request.url.path
            args = json.loads(request.content)
            in_flight[kind] += 1
            peak[kind] = max(peak[kind], in_flight[kind])
            if in_flight[kind] == (12 if kind == "/search" else 2):
                gates[kind].set()
            await gates[kind].wait()
            await asyncio.sleep(0)
            in_flight[kind] -= 1
            if kind == "/search":
                month = re.search(r"; (\d{4}-\d{2})\]", args["query"])[1]
                if month == "2026-04" and args["include_domains"] == ["stats.example"]:
                    return httpx.Response(503)
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {"url": f"https://stats.example/{month}", "title": f"Official {month}"},
                            {"url": "https://stats.example/archive", "title": "Official archive"},
                        ],
                        "usage": {"credits": 1},
                    },
                )
            url = args["urls"][0]
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": url,
                            "raw_content": "In May 2026 the official rate was 2.7 percent.",
                        }
                    ],
                    "usage": {"credits": 1},
                },
            )

        transport = httpx.MockTransport(respond)
        client = httpx.AsyncClient
        monkeypatch.setattr(
            retrieval.httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw)
        )

        available = adaptive.tools() + graph_tools(graph) + tools
        boundary = ResearchBoundary(
            budget, evidence.store, {t.name for t in available}, evidence=evidence
        )

        async def call(name, args):
            surface = {t.name: t for t in adaptive.surface(available)}
            assert name in surface
            action_id = f"{name}-test-{len(boundary.cache)}"

            async def dispatch(req):
                request = req.tool_call
                return ToolMessage(
                    content=await surface[name].ainvoke(request["args"]),
                    tool_call_id=request["id"],
                    name=name,
                )

            request = SimpleNamespace(tool_call={"name": name, "id": action_id, "args": args})
            async with asyncio.timeout(5):
                result = await boundary.awrap_tool_call(request, dispatch)
            return json.loads(result.content)

        result = await call(
            "search_monthly_records",
            {
                "record_query": "official monthly release",
                "question_part_ids": ["rates"],
                "include_domains": ["stats.example"],
            },
        )
        failed = next(t for t in result["targets"] if t["reference_month"] == "2026-04")
        assert failed["error"] == "tavily_http_503" and not failed["candidate_ids"]
        may = next(t for t in result["targets"] if t["reference_month"] == "2026-05")
        june = next(t for t in result["targets"] if t["reference_month"] == "2026-06")
        shared = may["candidate_ids"][1]
        reads = await call(
            "read_evidence_batch",
            {
                "reads": [
                    {"gap_id": may["gap_id"], "version_id": may["candidate_ids"][0]},
                    {"gap_id": may["gap_id"], "version_id": shared},
                    {"gap_id": june["gap_id"], "version_id": shared},
                ]
            },
        )
        assert len(reads["originals"]) == 2
        assert len(reads["originals"][1]["gap_ids"]) == 2
        pid = reads["originals"][0]["passage_ids"][0]
        bound = await call(
            "append_event_finding",
            {
                "node_id": "question",
                "expected_revision": adaptive.runtime_summary()["target_revision"],
                "finding": {
                    "assertion": "The official rate for May 2026 was 2.7 percent.",
                    "stance": "supports",
                    "occurrence": "observed",
                    "originals": [{"passage_id": pid}],
                    "question_part_id": "rates",
                    "event_time": {"precision": "month", "start": "2026-05"},
                    "monthly_value": 2.7,
                },
            },
        )
        assert "error" not in bound, bound
        closed = await call(
            "close_gap_round", {"reason": "Retain the supported value and disclose remaining gaps"}
        )
        assert closed["evidence_gain"] and not adaptive.active()
        assert sum(t["covered"] for t in closed["targets"]) == 1
        retry = pending_targets(adaptive)
        assert len(retry) == 11 and may["gap_id"] not in {g["gap_id"] for g in retry}
        feedback = adaptive.prompt()
        assert "tavily_http_503" in feedback and "previous_query" in feedback
        duplicate = await call(
            "search_monthly_records",
            {
                "record_query": "official monthly release",
                "question_part_ids": ["rates"],
                "include_domains": ["stats.example"],
            },
        )
        assert "change that target's evidence route" in duplicate["error"]
        alternate = await call(
            "search_monthly_records",
            {
                "record_query": "monthly statistical archive",
                "question_part_ids": ["rates"],
                "include_domains": ["archive.example"],
            },
        )
        april = next(t for t in alternate["targets"] if t["gap_id"] == failed["gap_id"])
        assert april["status"] == "returned" and april["candidate_ids"]
        assert sum(t["status"] == "not_dispatched" for t in alternate["targets"]) == 1

    asyncio.run(exercise())
    assert peak == {"/search": 12, "/extract": 2}
    assert budget.state["search_rounds"] == 22 and budget.state["tool_calls"] == 24
    assert not budget.state["retrieval_pending"]
    evidence.store.save("effective_config.json", {"workflow": "adaptive-event-map"})
    (tmp_path / "report.md").write_text("Saved evidence")
    view = build_snapshot(tmp_path)
    children = [
        n
        for n in view["nodes"]
        if n.get("label") == "按月并行检索" and n["id"].startswith("batch:round-1:")
    ]
    assert len(children) == 12 and all(len(n["requests"]) == 1 for n in children)
    assert max(n["started_sequence"] for n in children[:12]) < min(
        n["end_sequence"] for n in children
    )
    reads = [n for n in view["nodes"] if n.get("label") == "子线原文"]
    assert len(reads) == 2 and all(n["passage_ids"] for n in reads)
    join = next(n for n in view["nodes"] if n["title"].startswith("汇总子线结果"))
    assert {read["id"] for read in reads} <= {
        edge["from"] for edge in view["edges"] if edge["to"] == join["id"]
    }
    html = report_html(view)
    assert "+2.7" in html and "已搜待补证" in html


def test_shared_records_fan_out_by_month_with_42_in_flight(tmp_path):
    _, evidence, budget, graph, adaptive, begin = initialize(tmp_path)
    series = {"periods": 12, "anchor_part_id": "anchor", "unit": "%"}
    ids = ["annual", "monthly", "trade", "prices", "production"]
    args = [part("anchor"), *(part(pid, monthly_series=series) for pid in ids)]
    assert "42" in adaptive.prompt()
    assert "error" not in json.loads(asyncio.run(begin.ainvoke(plan(args))))
    ref = original(evidence, budget, "month", "The latest observation is October 2026.")
    bind(graph, ref, "anchor", month="2026-10")

    async def exercise():
        active = peak = 0
        gate = asyncio.Event()
        queries = []

        async def search(args, config):
            nonlocal active, peak
            queries.append(args["query"])
            active += 1
            peak = max(active, peak)
            if active == 42:
                gate.set()
            await gate.wait()
            await asyncio.sleep(0)
            active -= 1
            return json.dumps({"results": []})

        adaptive.search = SimpleNamespace(ainvoke=search)
        tool = next(t for t in adaptive.tools() if t.name == "search_monthly_records")
        async with asyncio.timeout(4):
            result = json.loads(
                await tool.ainvoke(
                    {
                        "record_query": "Official price release",
                        "question_part_ids": ["annual", "monthly"],
                        "additional_records": [
                            {"record_query": pid + " bulletin", "question_part_ids": [pid]}
                            for pid in ids[2:]
                        ],
                    }
                )
            )
        assert len(result["targets"]) == 60
        assert len(queries) == len(set(queries)) == 48
        assert peak == 42

    asyncio.run(exercise())


def test_evidence_plan_derives_dispatch_from_declared_cadence(tmp_path):
    _, evidence, budget, graph, adaptive, _ = initialize(tmp_path)
    # The fixture question is just “How did this event develop?”.
    # The source cadence comes exclusively from the planner's typed decision.
    available = graph_tools(graph) + adaptive.tools()
    tools = {tool.name: tool for tool in adaptive.surface(available)}
    assert set(tools) == {"begin_research_plan"}
    metadata = {
        "periods": 4,
        "publication": "Monthly transit bulletin",
        "domain": "transit.example",
        "unit": "passengers",
        "access_route": "Publisher releases or archived ridership table",
    }
    args = {
        "title": "Transit ridership",
        "scope": {
            **plan([])["scope"],
            "synthesis_tasks": ["Compare commuter and leisure use across all four releases"],
        },
        "monthly_measures": [
            {**metadata, "question": "Commuter traffic by month"},
            {**metadata, "question": "Leisure traffic by month"},
        ],
        "records": [],
    }
    result = asyncio.run(tools["begin_research_plan"].ainvoke(args))
    assert "error" not in json.loads(result)
    ref = original(
        evidence, budget, "release", "The latest published observation is February 2026."
    )
    graph.append(
        "investigation",
        Finding(
            assertion="Latest release",
            originals=[ref],
            stance="supports",
            occurrence="observed",
            question_part_id="latest-release",
            answers_question_part=True,
            event_time=TimeSpan(precision="month", start="2026-02"),
        ),
        1,
    )
    adaptive.refresh()
    assert [t.name for t in adaptive.surface(available)] == ["search_planned_months"]
    queries = []

    async def search(args, config):
        queries.append(args["query"])
        return '{"results":[]}'

    adaptive.search = SimpleNamespace(ainvoke=search)
    result = json.loads(
        asyncio.run(next(t for t in available if t.name == "search_planned_months").ainvoke({}))
    )
    assert len(result["targets"]) == 8 and len(queries) == 4
    assert {q.split("; ")[1][:7] for q in queries} == {"2025-11", "2025-12", "2026-01", "2026-02"}


def test_evidence_plan_preserves_one_off_records_and_synthesis(tmp_path):
    _, _, _, graph, adaptive, _ = initialize(tmp_path)
    tool = next(t for t in graph_tools(graph) if t.name == "begin_research_plan")
    inputs = [part("authorization"), part("completion")]
    synthesis = ["Compare the approval with the actual completion and cite both records."]
    args = {
        "title": "Reconstruct a transaction",
        "scope": {**plan([])["scope"], "synthesis_tasks": synthesis},
        "records": inputs,
        "monthly_measures": [],
    }
    assert "error" not in json.loads(asyncio.run(tool.ainvoke(args)))
    assert graph.summary()["scope"]["synthesis_tasks"] == synthesis
    adaptive.refresh()
    assert {g["question_part_id"] for g in pending_targets(adaptive)} == {
        "authorization",
        "completion",
    }
