import asyncio
import json

import pytest
from question_parts_fixture import bind, prepared
from test_event_graph import finding, original, record, setup

from news_agent.adaptive import AdaptiveInvestigation
from news_agent.budget import BudgetExceeded
from news_agent.event_graph import Finding, Scope, TimeSpan


class Search:
    def __init__(self, budget, fail=False):
        self.budget, self.fail, self.calls = budget, fail, []

    async def ainvoke(self, args, config):
        self.calls.append(args)
        self.budget.before_tool("search")
        if self.fail:
            raise RuntimeError("connection lost after dispatch")
        return json.dumps({"results": []})


def fixture(root, direction="backward"):
    _, e, b, g = setup(root)
    g.set_scope(
        Scope(
            direction=direction, anchor="Later outcome", objective="Check premise and development"
        ),
        0,
    )
    record(
        g,
        "premise",
        Finding(
            assertion="Did the event occur?",
            stance="unknown",
            occurrence="unknown",
            uncertainty="No original yet",
        ),
        kind="premise",
    )
    a = AdaptiveInvestigation(g, b)
    a.search = Search(b)
    b.state["search_round_limit"] = None
    return e, b, g, a


def gap(a, kind="missing_original", node="premise"):
    a.refresh()
    return next(
        r["gap_id"] for r in a.data["gaps"].values() if r["node_id"] == node and r["kind"] == kind
    )


def search(
    a, gid, query="new original", window=None, movement="unbounded", strategy="primary_record"
):
    return asyncio.run(
        a.search_gap(
            gid,
            query,
            strategy,
            window or TimeSpan(),
            movement,
            "Find original evidence for this exact node",
            {},
        )
    )


def test_reopening_same_delivered_url_is_blocked_across_rounds(tmp_path):
    _, _, _, a = fixture(tmp_path)
    search(a, gap(a))
    call = {
        "id": "original-open",
        "name": "open_url",
        "args": {"url": "https://issuer.example/release", "focus": " Latest statistics "},
    }
    assert a.before_tool(call) is None
    a.after_tool(call, json.dumps({"passages": [{"passage_id": "saved"}]}))
    a.close_round("End the first lookup")
    search(a, gap(a, "question_or_explanation_unresolved"), "different query")
    repeat = {
        **call,
        "id": "same-focus",
        "args": {**call["args"], "focus": "latest STATISTICS", "passage_limit": 8},
    }
    assert "already delivered" in a.before_tool(repeat)
    assert not a.active()["actions"]
    repeat["args"]["focus"] = "earlier benchmark"
    assert a.before_tool(repeat) is None


def test_targeted_range_stays_active_after_another_part_is_bound(tmp_path):
    _, evidence, budget, graph = prepared(tmp_path)
    a = evidence.adaptive
    queries = []

    class RetrievedCandidates:
        async def ainvoke(self, args, config):
            queries.append(args)
            return json.dumps({"results": [{"version_id": "history"}]})

    a.search = RetrievedCandidates()
    gid = gap(a, "question_or_explanation_unresolved", node="q")
    tool = next(t for t in a.tools() if t.name == "search_gap")
    result = json.loads(
        asyncio.run(
            tool.ainvoke(
                {
                    "gap_id": gid,
                    "question_part_id": "orbit",
                    "query": "orbit records May through October",
                    "strategy": "primary_record",
                    "window": {"precision": "range", "start": "2026-05", "end": "2026-10"},
                    "movement": "initial",
                    "reason": "Read records for the unresolved orbit part",
                }
            )
        )
    )
    assert result["question_part_id"] == "orbit"
    assert queries[0]["query"].endswith("[event date focus: 2026-05-01 to 2026-10-31]")

    launch = original(evidence, budget, "latest", "The probe launched on Oct. 2026.")
    bind(graph, launch, "launch", "observed", "The probe launched.")
    a.active()["local_no_gain"] = 2
    lookup = {
        "id": "read-trend",
        "name": "find_saved_text",
        "args": {"version_id": "history", "query": "orbit reference period"},
    }
    assert a.before_tool(lookup) is None
    a.prompt()
    assert a.active()["question_part_id"] == "orbit"
    assert a.measure(a.data["gaps"][gid], "orbit")["findings"] == []

    orbit = original(evidence, budget, "history", "The probe entered orbit.")
    a.after_tool(lookup, json.dumps({"passages": [{"passage_id": orbit.passage_id}]}))
    bind(graph, orbit, "orbit", "observed", "The probe entered orbit.")
    a.prompt()
    assert a.active() is None
    assert a.data["rounds"][-1]["retrieval_evidence_gain"]


def test_two_supplements_update_same_node_and_derive_resolution(tmp_path):
    e, b, g, a = fixture(tmp_path)
    gid = gap(a)
    search(a, gid, "early report")
    ref = original(e, b, "lead", "An early report raises a question but gives no answer.")
    record(
        g,
        "premise",
        Finding(
            assertion="Still unanswered", stance="context", occurrence="statement", originals=[ref]
        ),
        1,
        "premise",
    )
    first = a.close_round("Only context, no answer")
    assert first["gap_status"] == "resolved"  # missing original, not the user question
    unresolved = gap(a, "question_or_explanation_unresolved")
    # Context must not close an unresolved question just because it has an original.
    assert a.data["gaps"][unresolved]["status"] == "open"
    search(a, unresolved, "official later correction")
    ref = original(e, b, "official", "On November 2, 2026, Aurora confirmed the event.")
    record(
        g,
        "premise",
        finding(ref, "2026-11-02").model_copy(update={"answers_node_question": True}),
        2,
        "premise",
    )
    result = a.close_round("Original now directly addresses the premise")
    assert result["evidence_gain"] and result["gap_status"] == "resolved"
    assert len(g.data["nodes"]["premise"]["findings"]) == 3
    assert len(a.search.calls) == 2
    assert a.data["rounds"][0]["node_id"] == a.data["rounds"][1]["node_id"]


def test_primary_review_requires_exact_original_already_bound_to_target(tmp_path):
    e, b, g, a = fixture(tmp_path)
    ref = original(e, b, "issuer", "On November 1, 2026, Aurora issued its own decision.")
    with pytest.raises(ValueError, match="node-linked"):
        a.review_sources("premise", [ref], "issuer's own record")
    record(g, "premise", finding(ref), 1, "premise")
    gid = gap(a, "source_check")
    a.review_sources(
        "premise", [ref], "Attributed issuer statement, not an independent factual certification"
    )
    assert a.data["gaps"][gid]["status"] == "resolved"
    assert g.node_view("premise")["independence"] == "not_established"
    assert a.data["source_reviews"][0]["assessment"].startswith("model_assessed")


def test_two_low_yield_rounds_stall_then_new_evidence_reopens(tmp_path):
    e, b, g, a = fixture(tmp_path)
    gid = gap(a, "question_or_explanation_unresolved")
    for i in range(2):
        search(a, gid, f"different unproductive lead {i}")
        r = a.close_round("No useful original")
        assert not r["evidence_gain"]
    assert a.data["gaps"][gid]["status"] == "stalled"
    assert g.node_view("premise")["status"] == "important_unknown"
    with pytest.raises(ValueError, match="open gap"):
        search(a, gid, "third low yield")
    ref = original(e, b, "external", "Another material source raises the same question.")
    record(
        g,
        "premise",
        Finding(
            assertion="New context",
            stance="unknown",
            occurrence="statement",
            originals=[ref],
            uncertainty="Still no answer",
        ),
        1,
        "premise",
    )
    a.refresh()
    assert a.data["gaps"][gid]["status"] == "open"


def test_republications_do_not_count_as_independent_gain(tmp_path):
    e, b, g, a = fixture(tmp_path)
    refs = [
        original(e, b, vid, "On November 1, 2026, Aurora made its statement.")
        for vid in ["one", "copy", "copy2"]
    ]
    record(g, "premise", finding(refs[0]), 1, "premise")
    gid = gap(a, "source_check")
    for i in [1, 2]:
        search(a, gid, f"check copy {i}", strategy="independent")
        record(g, "premise", finding(refs[i]), i + 1, "premise")
        g.source_relation(
            "one",
            ["one", "copy", "copy2"][i],
            "same_origin",
            [refs[i]],
            "Source identifies itself as a copy",
        )
        a.refresh()
        assert not a.close_round("Same original, no independent evidence")["evidence_gain"]
    assert a.data["gaps"][gid]["status"] == "stalled"
    assert g.node_view("premise")["independence"] == "not_established"


@pytest.mark.parametrize(
    "movement,start,end",
    [
        ("expand_backward", "2026-09-01", "2026-11-01"),
        ("shift_backward", "2026-09-01", "2026-09-30"),
        ("expand_forward", "2026-10-01", "2026-12-01"),
        ("shift_forward", "2026-12-01", "2026-12-31"),
        ("narrow", "2026-10-10", "2026-10-20"),
    ],
)
def test_dynamic_windows_reach_actual_query_and_preserve_meaning(tmp_path, movement, start, end):
    _, _, _, a = fixture(tmp_path)
    gid = gap(a)
    search(
        a,
        gid,
        window=TimeSpan(precision="range", start="2026-10-01", end="2026-11-01"),
        movement="initial",
    )
    a.close_round("First window insufficient")
    search(
        a,
        gid,
        "different temporal lead",
        TimeSpan(precision="range", start=start, end=end),
        movement,
    )
    actual = a.search.calls[-1]["query"]
    assert start in actual and end in actual
    assert "include_domains" not in a.search.calls[-1]
    assert a.data["rounds"][-1]["movement"] == movement


def test_wrong_window_or_duplicate_query_never_dispatches(tmp_path):
    _, _, _, a = fixture(tmp_path)
    gid = gap(a)
    search(a, gid)
    with pytest.raises(ValueError, match="active round"):
        search(a, gid, "concurrent")
    a.close_round("No answer")
    with pytest.raises(ValueError, match="already attempted"):
        search(a, gid)
    with pytest.raises(ValueError, match="requires explicit"):
        search(a, gid, window=TimeSpan(precision="year", start="2025"), movement="shift_backward")
    assert len(a.search.calls) == 1


def test_unknown_time_can_be_anchored_then_focused(tmp_path):
    _, _, _, a = fixture(tmp_path)
    gid = gap(a)
    search(a, gid)
    a.close_round("Date lead discovered but not yet bound")
    search(a, gid, "dated original", TimeSpan(precision="day", start="2026-11-01"), "anchor")
    assert "2026-11-01" in a.search.calls[-1]["query"]


def test_uncertain_search_and_read_are_not_rebilled_after_resume(tmp_path):
    _, b, g, a = fixture(tmp_path)
    gid = gap(a)
    a.search.fail = True
    with pytest.raises(RuntimeError):
        search(a, gid)
    resumed = AdaptiveInvestigation(g, b)
    resumed.search = Search(b)
    assert resumed.data["rounds"][0]["status"] == "interrupted_unknown"
    with pytest.raises(ValueError, match="already attempted"):
        search(resumed, gid)
    search(resumed, gid, "new lead after uncertain request")
    call = {"name": "read_document", "id": "read-1", "args": {"version_id": "candidate"}}
    assert resumed.before_tool(call) is None
    again = AdaptiveInvestigation(g, b)
    assert "uncertain" in again.before_tool({**call, "id": "read-2"})
    assert len(resumed.search.calls) == 1


def test_search_round_cap_removed_only_for_adaptive_total_budget_remains(tmp_path):
    _, b, _, a = fixture(tmp_path)
    for _ in range(7):
        b.before_tool("search")
    assert b.state["search_rounds"] == 7
    b.state["tool_calls"] = b.config.max_tool_calls - b.config.reserve_tool_calls
    with pytest.raises(BudgetExceeded, match="research_finished"):
        b.before_tool("search")
    with pytest.raises(ValueError, match="research_finished"):
        search(a, gap(a))


def test_all_supplementary_reads_require_active_gap(tmp_path):
    _, _, _, a = fixture(tmp_path)
    assert "search_gap" in a.before_tool({"name": "search", "id": "bad", "args": {}})
    assert "start search_gap" in a.before_tool({"name": "read_document", "id": "bad2", "args": {}})


def test_fallback_preserves_graph_originals_and_unknowns_without_claim_records(tmp_path):
    e, b, g, a = fixture(tmp_path)
    ref = original(e, b, "one", "On November 1, 2026, Aurora made a statement.")
    record(
        g,
        "premise",
        Finding(
            assertion="A source statement, still no answer",
            stance="context",
            occurrence="statement",
            originals=[ref],
        ),
        1,
        "premise",
    )
    report = a.fallback_report("provider_timeout")
    assert ref.passage_id in report and "provider_timeout" in report
    assert "question_or_explanation_unresolved" in report


def test_original_binding_auto_closes_round_before_next_decision(tmp_path):
    e, b, g, a = fixture(tmp_path)
    gid = gap(a, "question_or_explanation_unresolved")
    search(a, gid)
    ref = original(e, b, "one", "On November 1, 2026, Aurora confirmed the event.")
    record(
        g, "premise", finding(ref).model_copy(update={"answers_node_question": True}), 1, "premise"
    )
    a.prompt()
    assert a.active() is None
    assert a.data["rounds"][0]["evidence_gain"]
    assert a.data["gaps"][gid]["status"] == "resolved"


def test_bare_ids_linked_once_without_inventing_new_ids(tmp_path):
    _, _, _, a = fixture(tmp_path)
    text = "source p-0123456789abcdef and [p-0123456789abcdef]"
    assert a.link_citations(text) == "source [p-0123456789abcdef] and [p-0123456789abcdef]"


@pytest.mark.parametrize(
    "quote",
    [
        "During November 2026, Aurora issued its statement.",
        "During Nov 2026, Aurora issued its statement.",
        "2026年11月，Aurora发布声明。",
    ],
)
def test_explicit_original_month_is_accepted_but_another_year_is_not(tmp_path, quote):
    e, b, g, _ = fixture(tmp_path)
    ref = original(e, b, "one", quote)
    f = Finding(
        assertion="Source dates the statement to a month",
        stance="supports",
        occurrence="statement",
        originals=[ref],
        event_time=TimeSpan(precision="month", start="2026-11"),
    )
    record(g, "dated", f)
    with pytest.raises(ValueError, match="month requires"):
        record(
            g,
            "wrong",
            f.model_copy(update={"event_time": TimeSpan(precision="month", start="2025-11")}),
        )
