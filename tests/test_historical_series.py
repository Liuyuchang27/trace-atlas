"""Exercise live-series orchestration and chart binding without paying for retrieval."""

import asyncio
import json
from datetime import UTC, datetime

import pytest
from test_monthly_series import begin_parts

from news_agent.event_graph import Finding
from news_agent.event_report import choices
from news_agent.historical_series import (
    FRED_HOST,
    RepositorySeriesLinks,
    extract_monthly_changes,
    normalize_title_terms,
    publisher_matches,
)
from news_agent.live import question_task
from news_agent.schemas import DocumentVersion
from news_agent.series_collection import monthly_history_tools
from news_agent.temporal_review import fixture_review, review_input, validate_review
from news_agent.web_view import build_snapshot

LANDING = f"https://{FRED_HOST}/series/EXAMPLE1"
TABLE = f"https://{FRED_HOST}/data/EXAMPLE1"
METADATA = """| Title | Example survey households |
| Series ID | EXAMPLE1 |
| Source | Example Statistical Office |
| Seasonal Adjustment | Seasonally Adjusted |
| Frequency | Monthly |
| Units | Thousands of Persons |
| Date Range | 2025-08-01 to 2026-02-01 |"""
ROWS = (
    "| 2025-08-01 | 200 |\n"
    "| 2025-09-01 | 201.3 |\n"
    "#2025-10-01|202 #2025-11-01|200 #2025-12-01|200 #2026-01-01|208 #2026-02-01|211"
)


def document(url=TABLE, content=METADATA + "\n\n" + ROWS):
    return DocumentVersion(
        version_id="data" if url == TABLE else "landing",
        title="Example survey households",
        url=url,
        content=content,
        retrieved_at=datetime(2026, 3, 15, tzinfo=UTC),
    )


@pytest.mark.parametrize("catalog_available", [False, True])
def test_repository_fallback_binds_six_calculated_months_and_renders_citations(
    tmp_path, monkeypatch, catalog_available
):
    evidence, budget, graph = begin_parts(
        tmp_path,
        [
            {"part_id": "latest", "question": "Latest change?", "required_occurrence": "observed"},
            {
                "part_id": "trend",
                "question": "Monthly trend?",
                "requires_date": True,
                "monthly_series": {"periods": 6, "anchor_part_id": "latest", "unit": "人"},
            },
        ],
    )
    adaptive = evidence.adaptive
    adaptive.monthly_history_enabled = True
    calls = []

    hit = {"title": "Example survey households", "url": LANDING, "version_id": "landing"}

    async def catalog(query, corpus, budget, store):
        calls.append((query, "catalog"))
        return [hit] if catalog_available else []

    monkeypatch.setattr("news_agent.series_collection.search_series_catalog", catalog)

    class Repository:
        source_metadata = {"data": {"truncated": False}}

        async def asearch(self, query, limit, timeout, include_domains):
            calls.append((query, include_domains))
            return [hit] if include_domains == [FRED_HOST] else []

        async def aread(self, version_id, timeout):
            calls.append(version_id)
            return (
                document(LANDING, "[Full table](/data/EXAMPLE1)")
                if version_id == "landing"
                else document()
            )

        def register(self, url, title):
            assert url == TABLE
            return "data"

    tool, select = monthly_history_tools(adaptive, Repository())
    assert adaptive.surface([tool, select]) == [tool]
    receipt = json.loads(
        asyncio.run(
            tool.ainvoke(
                {
                    "query": "official monthly survey households historical levels",
                    "title_terms": [
                        "survey",
                        "households",
                        "monthly levels",
                        "seasonally adjusted",
                    ],
                    "publisher_terms": ["Example Statistical Office", "ESO"],
                    "publisher_domain": "statistics.example",
                    "method": "monthly_net_change",
                    "adjustment": "seasonally_adjusted",
                }
            )
        )
    )
    assert receipt["status"] == "choose_source", receipt
    assert adaptive.surface([tool, select]) == [select]
    assert calls[-1] != "landing"
    receipt = json.loads(
        asyncio.run(
            select.ainvoke(
                {
                    "version_id": receipt["candidates"][0]["version_id"],
                    "reason": "Matches the survey",
                }
            )
        )
    )
    assert receipt["status"] == "completed", receipt
    assert calls[0][1] == ["statistics.example"]
    assert calls[1] == ("survey", "catalog")
    assert calls[-2:] == ["landing", "data"]
    if not catalog_available:
        assert calls[2][1] == [FRED_HOST]
    assert budget.state["search_rounds"] == 2 + (not catalog_available)
    assert budget.state["tool_calls"] == 4 + (not catalog_available)
    assert adaptive.pending_series() is None
    trend = graph.node_view("stats")["part_status"][-1]["month_coverage"]
    assert trend["anchor"] == "2026-02" and not trend["missing"]
    assert [p["value"] for p in trend["points"].values()] == [1300, 700, -2000, 0, 8000, 3000]
    for row in graph.data["nodes"]["stats"]["findings"][1:]:
        assert {ref["version_id"] for ref in row["references"]} == {"data"}
        assert len(row["references"]) == 3
    visible = {p.passage_id: p.text for p in evidence.passages.values()}
    evidence.event_graph = graph
    plan = choices(evidence, {"documents": [{"excerpts": visible}]})
    review = review_input(question_task("Monthly change trend"), evidence, plan)
    assert len(review["question_parts"][-1]["assigned_finding_ids"]) == 6
    response = fixture_review(review)
    for row in response["decisions"]:
        row.update(event_verdict="needs_evidence", date_role="unresolved")
    reviewed = validate_review(json.dumps(response), review)
    assert len(reviewed) == 6 and all(r["event_verdict"] == "supported" for r in reviewed)
    payload = graph.data["nodes"]["stats"]["findings"][-1]["finding"].copy()
    payload["monthly_value"] = 1300
    with pytest.raises(ValueError, match="reproducible adjacent levels"):
        graph.append("stats", Finding.model_validate(payload), 7)
    evidence.store.save("effective_config.json", {"workflow": "adaptive-event-map"})
    (tmp_path / "report.md").write_text("Saved report")
    view = build_snapshot(tmp_path)
    assert view["report_brief"]["charts"][0]["source"] == TABLE
    assert view["report_brief_html"].count("series-bar") == 6
    assert view["report_brief_html"].count("上月原值") == 6
    assert any(n["title"] == "当前来源缺少完整表，改查统计数据库" for n in view["nodes"])


def test_catalog_only_accepts_series_links_and_reads_nested_titles():
    parser = RepositorySeriesLinks()
    parser.feed(
        '<a href="/graph?g=abc">Chart</a>'
        '<a href="/series/EXAMPLE1">Example <strong>survey</strong> households</a>'
        '<a href="https://example.org/series/OTHER">Other site</a>'
    )
    assert parser.links == {LANDING: "Example survey households"}


def test_publisher_accepts_spelled_out_name_and_acronym():
    name = "U.S. Bureau of Labor Statistics"
    assert publisher_matches(name, ["Bureau of Labor Statistics", "BLS"])
    assert not publisher_matches(name, ["Bureau of Labor Statistics", "BEA"])
    assert not publisher_matches("Private Statistical Bureau", ["BLS"])


def test_title_modifiers_are_checked_in_the_correct_metadata_field():
    terms, adjustment = normalize_title_terms(
        ["Example households", "not seasonally adjusted"], "any"
    )
    assert terms == ["Example households"] and adjustment == "not_seasonally_adjusted"
    with pytest.raises(ValueError, match="季调口径冲突"):
        normalize_title_terms(["Example", "not seasonally adjusted"], "seasonally_adjusted")


@pytest.mark.parametrize(
    "modified",
    [
        METADATA.replace("Thousands of Persons", "Index") + "\n\n" + ROWS,
        METADATA.replace("Seasonally Adjusted", "Not Seasonally Adjusted") + "\n\n" + ROWS,
        METADATA + "\n\n" + ROWS.replace("#2025-10-01|202 ", ""),
        METADATA + "\n\n" + ROWS + "\n#2026-02-01|212",
        METADATA + "\n\n" + ROWS.replace(" #2026-02-01|211", ""),
    ],
)
def test_incompatible_incomplete_or_conflicting_history_fails_closed(modified):
    with pytest.raises(ValueError):
        extract_monthly_changes(
            document(content=modified),
            {"periods": 6, "unit": "人"},
            ["survey", "households"],
            ["Example Statistical Office"],
            "seasonally_adjusted",
        )
