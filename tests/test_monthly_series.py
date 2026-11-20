"""The requested period is checked against the latest observed month and saved originals."""

import asyncio
import json

import pytest
from test_event_graph import original, setup

from news_agent.adaptive import AdaptiveInvestigation
from news_agent.event_graph import Finding, TimeSpan, graph_tools
from news_agent.monthly_series import explicit_search_months, monthly_value_in_original
from news_agent.web_view import build_snapshot, report_html


def begin_parts(tmp_path, parts):
    _, evidence, budget, graph = setup(tmp_path)
    evidence.adaptive = AdaptiveInvestigation(graph, budget)
    begin = next(t for t in graph_tools(graph) if t.name == "begin_investigation")
    begun = json.loads(
        asyncio.run(
            begin.ainvoke(
                {
                    "node_id": "stats",
                    "title": "Monthly totals",
                    "scope": {"direction": "backward", "anchor": "Latest", "objective": "Trend"},
                    "question_parts": parts,
                }
            )
        )
    )
    assert "error" not in begun
    return evidence, budget, graph


def test_in_thousands_caption_is_applied_to_the_correct_reference_column():
    table = (
        "| Category | Aug. 2026 | Sept. 2026 | Oct. 2026 |\n"
        "| Employment (over-the-month change, in thousands) |\n"
        "| Total nonfarm | 186 | 60 | 165 |"
    )
    assert monthly_value_in_original(165000, "2026-10", [table])
    assert not monthly_value_in_original(186000, "2026-10", [table])
    assert not monthly_value_in_original(165000, "2026-10", [table.replace(", in thousands", "")])
    assert explicit_search_months("May June July August September October 2026") == {
        f"2026-{m:02d}" for m in range(5, 11)
    }


def test_series_requires_all_requested_months_and_exports_sourced_bars(tmp_path):
    evidence, budget, graph = begin_parts(
        tmp_path,
        [
            {"part_id": "latest", "question": "Latest monthly count?"},
            {
                "part_id": "trend",
                "question": "Monthly trend?",
                "requires_date": True,
                "monthly_series": {"periods": 3, "anchor_part_id": "latest", "unit": "件"},
            },
        ],
    )

    def append(when, part, value, text):
        ref = original(evidence, budget, "record-" + when, text)
        graph.append(
            "stats",
            Finding(
                assertion="The original contains the monthly count.",
                stance="supports",
                occurrence="observed",
                event_time=TimeSpan(precision="month", start=when),
                originals=[ref],
                answers_question_part=part == "latest",
                question_part_id=part,
                monthly_value=value,
            ),
            len(graph.data["nodes"]["stats"]["findings"]),
        )

    append("2026-01", "latest", 17000, "The count rose by 17,000 in January 2026.")
    with pytest.raises(ValueError, match="original number"):
        append("2025-12", "trend", 2000, "| Dec. 2025 | -2,000 |")
    trend = graph.node_view("stats")["part_status"][-1]
    assert trend["month_coverage"]["missing"] == ["2025-11", "2025-12"]
    assert trend["status"] == "unresolved"
    adaptive = evidence.adaptive
    adaptive.refresh()
    gap = next(g for g in adaptive.data["gaps"].values() if g.get("reference_month") == "2025-11")
    december = next(
        g for g in adaptive.data["gaps"].values() if g.get("reference_month") == "2025-12"
    )
    dispatched = []

    class History:
        async def ainvoke(self, args, config):
            dispatched.append(args["query"])
            return json.dumps({"results": [{"version_id": "historical"}]})

    adaptive.search = History()
    tool = next(t for t in adaptive.tools() if t.name == "search_gap")

    def search(target, query="official monthly totals"):
        args = {
            "gap_id": target["gap_id"],
            "query": query,
            "strategy": "primary_record",
            "reason": "Find the missing month",
        }
        action_id = f"search-{evidence.store.sequence + 1}"
        evidence.store.event(
            "tool_dispatch", name="search_gap", action_id=action_id, arguments=args
        )
        budget.state["active_tool_action_id"] = action_id
        return json.loads(asyncio.run(tool.ainvoke(args)))

    assert "one reference month" in search(gap, "2025-11 to 2025-12")["error"]
    assert not dispatched
    assert search(gap)["reference_month"] == "2025-11"
    row = adaptive.active()
    row["actions"].append({"action_id": "read", "passage_ids": ["from-history"]})
    row["local_no_gain"] = 2
    call = {
        "name": "append_event_finding",
        "id": "rejected",
        "args": {
            "finding": {"question_part_id": "trend"},
        },
    }
    for turn in (1, 1, 2):
        budget.state["model_calls"] = turn
        adaptive.after_tool(call, '{"error":"monthly value needs original number"}')
    receipt = json.loads(adaptive.prompt().split("\n", 1)[1])
    assert adaptive.active() is None
    assert receipt["monthly_source_replan"]["gap_id"] == gap["gap_id"]
    assert gap["status"] == "open"
    assert "replan_gap" in search(december)["error"]
    evidence.store.save("effective_config.json", {"workflow": "adaptive-event-map"})
    (tmp_path / "report.md").write_text("Saved report")
    partial = build_snapshot(tmp_path)["report_brief_html"]
    assert "已搜待补证" in partial and "未按月检索" in partial
    replan = next(t for t in adaptive.tools() if t.name == "replan_gap")
    plan = json.loads(
        asyncio.run(
            replan.ainvoke(
                {
                    "gap_id": gap["gap_id"],
                    "route": "change_route",
                    "reason": "Check a different authority",
                    "source_domains": ["statistics.example"],
                }
            )
        )
    )
    assert plan["status"] == "decided"
    assert "untried" in search(gap)["error"]
    assert search(december)["reference_month"] == "2025-12"
    assert "November 2025" in dispatched[0] and "December 2025" in dispatched[1]
    append("2025-12", "trend", -2000, "| Dec. 2025 | -2,000 |")
    append("2025-11", "trend", 0, "| Nov 2025 | 0 |")
    adaptive.prompt()
    assert adaptive.active() is None
    assert gap["status"] == december["status"] == "resolved"
    assert graph.node_view("stats")["part_status"][-1]["status"] == "model_assessed_answered"
    evidence.store.save("effective_config.json", {"workflow": "adaptive-event-map"})
    (tmp_path / "report.md").write_text("Saved report")
    view = build_snapshot(tmp_path)
    months = [n for n in view["nodes"] if n.get("label") == "按月补查"]
    assert [n["details"]["target"]["reference_month"] for n in months] == ["2025-11", "2025-12"]
    chart = view["report_brief"]["charts"][0]
    assert [p["value"] for p in chart["points"]] == [0, -2000, 17000]
    exported = report_html(view)
    assert "series-bar down" in exported
    assert all(f'id="{m["passages"][0]}"' in exported for m in chart["points"])


def test_non_monthly_parts_are_searched_and_resolved_independently(tmp_path):
    evidence, budget, graph = begin_parts(
        tmp_path,
        [
            {"part_id": "north", "question": "What did the northern region decide?"},
            {"part_id": "south", "question": "What did the southern region decide?"},
        ],
    )
    adaptive = evidence.adaptive
    targets = {
        g["question_part_id"]: g for g in adaptive.summary()["queue"] if g.get("question_part_id")
    }
    assert targets.keys() == {"north", "south"}

    class EmptyResults:
        async def ainvoke(self, args, config):
            return '{"results":[]}'

    adaptive.search = EmptyResults()
    for part in ("north", "south"):
        result = asyncio.run(
            adaptive.search_gap(
                targets[part]["gap_id"],
                f"{part} region original decision",
                "primary_record",
                TimeSpan(),
                "unbounded",
                "Find the decision for this region",
                {},
            )
        )
        assert result["question_part_id"] == part
        adaptive.prompt()
    ref = original(evidence, budget, "south", "The southern authority issued its decision.")
    graph.append(
        "stats",
        Finding(
            assertion="The southern authority issued a decision.",
            stance="supports",
            occurrence="observed",
            originals=[ref],
            question_part_id="south",
            answers_question_part=True,
        ),
        1,
    )
    adaptive.refresh()
    assert adaptive.data["gaps"][targets["south"]["gap_id"]]["status"] == "resolved"
    assert adaptive.data["gaps"][targets["north"]["gap_id"]]["status"] == "open"
