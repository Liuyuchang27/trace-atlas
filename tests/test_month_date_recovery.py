"""Exercise dated context and explicit unknowns through the graph tools."""

import asyncio
import json
from datetime import UTC, datetime

import pytest
from test_adaptive import fixture
from test_event_graph import original

from news_agent.budget import Budget
from news_agent.event_graph import Finding, TimeSpan, graph_tools
from news_agent.event_report import precision_issue
from news_agent.harness import remember_passages
from news_agent.schemas import DocumentVersion


def test_saved_table_month_abbreviation_passes_both_date_guards(tmp_path):
    e, b, g, a = fixture(tmp_path)
    e.adaptive = a
    ref = original(e, b, "series", "| Jun. 2026 | 31,000 |")
    finding = Finding(
        assertion="2026年6月的月度值见表。",
        stance="supports",
        occurrence="observed",
        event_time=TimeSpan(precision="month", start="2026-06"),
        originals=[ref],
    )
    assert precision_issue(finding, e) is None
    g.append("premise", finding, 1)
    assert g.data["nodes"]["premise"]["findings"][-1]["finding"]["event_time"]["start"] == "2026-06"


@pytest.mark.parametrize("deliver_header", [True, False])
def test_month_write_uses_only_explicit_delivered_date_support(tmp_path, deliver_header):
    evidence, budget, graph, adaptive = fixture(tmp_path)
    evidence.adaptive = adaptive
    header, body, publication = evidence.add_document(
        DocumentVersion(
            version_id="series",
            title="Monthly release",
            url="https://issuer.example/record",
            content=(
                "MONTHLY RESULTS — OCTOBER 2026\n\n"
                "The number of openings rose in October.\n\nPublished: October 12, 2026"
            ),
            published_at=datetime(2026, 10, 12, tzinfo=UTC),
            retrieved_at=datetime(2026, 11, 19, tzinfo=UTC),
        )
    )
    original(evidence, budget, "other", "A different series published results for October 2026.")
    remember_passages(budget, [body.passage_id, publication.passage_id])
    if deliver_header:
        remember_passages(budget, [header.passage_id])
    tool = next(t for t in graph_tools(graph) if t.name == "append_event_finding")
    args = {
        "node_id": "premise",
        "expected_revision": 1,
        "finding": {
            "assertion": "Openings rose in October 2026.",
            "stance": "supports",
            "occurrence": "observed",
            "event_time": {"precision": "month", "start": "2026-10"},
            "originals": [{"passage_id": body.passage_id}],
        },
    }
    rejected = json.loads(asyncio.run(tool.ainvoke(args)))
    assert "month requires ISO month" in rejected["error"]
    correction = rejected["date_correction"]
    assert budget.state["last_finding_rejection"]["date_correction"] == correction
    adaptive.write_recovery.capture({"id": "initial", "name": tool.name, "args": args}, rejected)
    assert len(graph.data["nodes"]["premise"]["findings"]) == 1
    graph.budget = Budget(budget.config, budget.store)
    args["finding"]["assertion"] = "In October 2026, the number of openings rose."
    repeat = json.loads(asyncio.run(tool.ainvoke(args)))
    assert repeat["error"] == "repeated unsupported date"
    assert len(graph.data["nodes"]["premise"]["findings"]) == 1
    if deliver_header:
        assert [r["passage_id"] for r in correction["date_support_candidates"]] == [
            header.passage_id
        ]
        assert adaptive.write_recovery.proposal is None
        args["finding"]["originals"].append({"passage_id": header.passage_id})
        assert "event_graph" in json.loads(asyncio.run(tool.ainvoke(args)))
        row = graph.data["nodes"]["premise"]["findings"][-1]
        assert row["finding"]["event_time"]["start"] == "2026-10"
        assert row["references"][-1]["quote"] == header.text
    else:
        assert "date_support_candidates" not in correction
        assert adaptive.write_recovery.pending()
        undated = next(t for t in graph_tools(graph) if t.name == "append_undated_finding")
        args["finding"].pop("event_time")
        args["finding"].update(
            assertion="Openings rose in October.",
            date_text="October",
            uncertainty="The original paragraph does not identify the year.",
        )
        assert "event_graph" in json.loads(asyncio.run(undated.ainvoke(args)))
        row = graph.data["nodes"]["premise"]["findings"][-1]
        assert row["finding"]["event_time"]["precision"] == "unknown"
        assert row["finding"]["event_time"]["start"] is None
    assert budget.state["model_calls"] == 0
