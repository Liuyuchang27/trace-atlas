import asyncio
import copy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from news_agent.budget import Budget
from news_agent.event_graph import EventGraph, Finding, OriginalRef, Scope, TimeSpan, graph_tools
from news_agent.harness import remember_passages
from news_agent.schemas import BudgetConfig, DocumentVersion, ResearchTask
from news_agent.storage import EvidenceStore, RunStore, read_json


def setup(root):
    task = ResearchTask(
        task_id="generic-event",
        event="Aurora public event",
        question="How did this event develop?",
        mode="retrospective",
        synthetic=True,
    )
    e = EvidenceStore(RunStore(root))
    b = Budget(BudgetConfig(), e.store)
    return task, e, b, EventGraph(task, e, b)


def original(e, b, vid, text, delivered=True):
    p = e.add_document(
        DocumentVersion(
            version_id=vid,
            title=vid,
            url=f"https://{vid}.example/record",
            content=text,
            published_at=datetime(2026, 11, 10, tzinfo=UTC),
            retrieved_at=datetime(2026, 11, 15, tzinfo=UTC),
        )
    )[0]
    if delivered:
        remember_passages(b, [p.passage_id])
    return OriginalRef(passage_id=p.passage_id, quote=p.text)


def finding(ref, when="2026-11-01", **extra):
    return Finding(
        assertion="The source states an event.",
        stance="supports",
        occurrence="statement",
        originals=[ref],
        event_time=TimeSpan(precision="day", start=when),
        **extra,
    )


def record(g, nid, f, revision=0, kind="statement"):
    return g.record(nid, nid, "Aurora", kind, "major", "Changes the user answer", f, revision)


@pytest.mark.parametrize("direction", ["backward", "forward", "both"])
def test_scope_direction_is_versioned_plan_not_asserted_premise(tmp_path, direction):
    _, _, _, g = setup(tmp_path)
    s = Scope(
        direction=direction,
        anchor="User gives an outcome",
        objective="Test premise and trace origin",
    )
    g.set_scope(s, 0)
    assert g.summary()["scope"]["premise"] == "untested"
    g.set_scope(s.model_copy(update={"premise": "not_established"}), 1)
    assert len(g.data["scope_history"]) == 2
    assert g.data["scope_history"][0]["scope"]["premise"] == "untested"
    before = copy.deepcopy(g.data)
    with pytest.raises(ValueError, match="stale"):
        g.set_scope(s, 0)
    assert g.data == before


@pytest.mark.parametrize(
    "span",
    [
        {"precision": "unknown", "start": "2026-11-01"},
        {"precision": "range", "start": "2026-11-02", "end": "2026-11-01"},
        {"precision": "month", "start": "2026-13"},
        {"precision": "day", "start": "2026-11-31"},
    ],
)
def test_invalid_time_precision_or_ranges_are_rejected(span):
    with pytest.raises(ValidationError):
        TimeSpan(**span)


def test_original_offsets_and_three_clocks_stay_distinct(tmp_path):
    _, e, b, g = setup(tmp_path)
    ref = original(e, b, "one", "On November 1, 2026, Aurora announced the decision.")
    record(g, "decision", finding(ref))
    row = g.data["nodes"]["decision"]["findings"][0]
    assert row["finding"]["event_time"]["start"] == "2026-11-01"
    assert row["references"][0]["published_at"].startswith("2026-11-10")
    assert row["references"][0]["retrieved_at"].startswith("2026-11-15")
    bound = row["references"][0]
    assert e.documents["one"].content[bound["start"] : bound["end"]] == ref.quote
    with pytest.raises(ValueError, match="not metadata"):
        record(g, "wrong", finding(ref, "2026-11-10"))
    view = g.node_view("decision", "2026-11-01")
    assert view["historical_availability"] == "not_established_by_this_view"


def test_append_revision_dedup_stale_and_identity_are_atomic(tmp_path):
    _, e, b, g = setup(tmp_path)
    a = original(e, b, "one", "On November 1, 2026, Aurora announced the decision.")
    f = finding(a)
    record(g, "decision", f)
    before = (tmp_path / "event_graph.json").read_bytes()
    record(g, "decision", f)  # exact idempotent retry
    assert (tmp_path / "event_graph.json").read_bytes() == before
    other = f.model_copy(update={"assertion": "Another source assertion"})
    with pytest.raises(ValueError, match="stale"):
        record(g, "decision", other)
    with pytest.raises(ValueError, match="immutable"):
        g.record("decision", "Different event", "Aurora", "statement", "major", "Reason", other, 1)
    record(g, "decision", other, revision=1)
    assert len(g.data["nodes"]["decision"]["findings"]) == 2
    assert g.data["nodes"]["decision"]["findings"][0]["finding"] == f.model_dump()


def test_unread_or_edited_quote_does_not_mutate_graph(tmp_path):
    _, e, b, g = setup(tmp_path)
    ref = original(e, b, "one", "On November 1, 2026, Aurora announced the decision.", False)
    before = copy.deepcopy(g.data)
    with pytest.raises(ValueError, match="tool-delivered"):
        record(g, "one", finding(ref))
    assert g.data == before
    remember_passages(b, [ref.passage_id])
    with pytest.raises(ValueError, match="does not match"):
        record(g, "one", finding(ref.model_copy(update={"quote": "Made up original sentence"})))
    assert g.data == before


def test_republication_family_is_one_source_and_domains_do_not_prove_independence(tmp_path):
    _, e, b, g = setup(tmp_path)
    a = original(e, b, "issuer", "On November 1, 2026, Aurora announced the decision.")
    z = original(
        e,
        b,
        "repost",
        "On November 1, 2026, Aurora announced the decision. Republished from issuer.",
    )
    record(g, "one", finding(a))
    record(g, "one", finding(z), 1)
    assert g.node_view("one")["independence"] == "not_established"
    with pytest.raises(ValueError, match="not domains"):
        g.source_relation("issuer", "repost", "distinct_origin", [], "Different domains")
    g.source_relation("issuer", "repost", "same_origin", [z], "Explicit republication statement")
    assert len(g.node_view("one")["source_families"]) == 1
    assert g.node_view("one")["independence"] == "not_established"


def test_later_correction_preserves_earlier_cutoff_state_and_originals(tmp_path):
    _, e, b, g = setup(tmp_path)
    a = original(e, b, "early", "On November 1, 2026, Aurora said all batches were included.")
    z = original(
        e,
        b,
        "later",
        "On November 2, 2026, Aurora corrected its earlier statement: "
        "only some batches were included.",
    )
    record(g, "early", finding(a))
    record(g, "later", finding(z, "2026-11-02"))
    g.relate("later", "early", "corrects", [z], "Source explicitly corrects the earlier scope")
    assert g.node_view("early")["status"] == "corrected"
    assert g.node_view("early", "2026-11-01")["status"] == "source_supported"
    assert not g.node_view("early", "2026-11-01")["annotations"][0]["active_at_cutoff"]
    assert g.node_view("later", "2026-11-01")["status"] == "later_context"
    assert len(g.data["nodes"]["early"]["findings"]) == 1


def test_nonoverlapping_applicability_is_change_not_same_time_contradiction(tmp_path):
    _, e, b, g = setup(tmp_path)
    a = original(e, b, "early", "On November 1, 2026, Aurora's role was active.")
    z = original(e, b, "later", "On November 2, 2026, Aurora's role was inactive.")
    x = finding(a, valid_time=TimeSpan(precision="day", start="2026-11-01"))
    y = finding(
        z, "2026-11-02", valid_time=TimeSpan(precision="day", start="2026-11-02")
    ).model_copy(update={"stance": "contradicts"})
    record(g, "role", x)
    record(g, "role", y, 1)
    assert g.node_view("role")["status"] == "changes_over_time"
    assert g.node_view("role", "2026-11-01")["status"] == "source_supported"
    assert g.node_view("role", "2026-11-03")["status"] == "outside_applicability"
    record(g, "new_role", y)
    with pytest.raises(ValueError, match="same-time"):
        g.relate("new_role", "role", "contradicts", [z], "Different dates", "different_period")


def test_correlation_does_not_become_cause_and_important_unknown_can_stay(tmp_path):
    _, e, b, g = setup(tmp_path)
    ref = original(
        e, b, "one", "On November 1, 2026, the supplier changed and a fault was reported."
    )
    record(g, "supplier", finding(ref))
    record(g, "fault", finding(ref))
    before = copy.deepcopy(g.data)
    with pytest.raises(ValueError, match="causal source"):
        g.relate("supplier", "fault", "cause_stated", [ref], "Time proximity")
    assert g.data == before
    g.relate(
        "supplier",
        "fault",
        "cause_hypothesis",
        [ref],
        "Possible explanation; direct evidence missing",
    )
    assert g.data["relations"][0]["verification"] == "hypothesis"
    record(
        g,
        "why",
        Finding(
            assertion="Did supplier change contribute?",
            stance="unknown",
            occurrence="unknown",
            uncertainty="Need a direct technical record",
        ),
        kind="question",
    )
    assert g.node_view("why")["status"] == "important_unknown"
    assert g.node_view("why")["findings"][0]["finding"]["event_time"]["precision"] == "unknown"


def test_checkpoint_replays_missing_journal_event_once_without_model_or_network(tmp_path):
    task, e, b, g = setup(tmp_path)
    a = original(e, b, "one", "On November 1, 2026, Aurora announced the decision.")
    record(g, "one", finding(a))
    journal = tmp_path / "trajectory.jsonl"
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines[:-1]) + "\n")
    state = copy.deepcopy(b.state)
    restored = EventGraph(task, e, b)
    assert restored.data == g.data and b.state == state and b.state["model_calls"] == 0
    after = journal.read_text()
    EventGraph(task, e, b)
    assert journal.read_text() == after


def test_tool_inspection_and_scope_return_current_state_with_no_extra_model_call(tmp_path):
    _, _, b, g = setup(tmp_path)
    tools = {t.name: t for t in graph_tools(g)}
    result = asyncio.run(
        tools["set_investigation_scope"].ainvoke(
            {
                "scope": {
                    "direction": "both",
                    "anchor": "Unconfirmed premise",
                    "objective": "Verify event and trace both directions",
                }
            }
        )
    )
    assert '"direction": "both"' in result
    assert b.state["model_calls"] == 0
    assert read_json(tmp_path / "event_graph.json")["scope_history"]


def test_new_original_updates_same_question_and_clears_only_its_missing_evidence_gap(tmp_path):
    _, e, b, g = setup(tmp_path)
    record(
        g,
        "question",
        Finding(
            assertion="What happened?",
            stance="unknown",
            occurrence="unknown",
            uncertainty="Need the original announcement",
        ),
        kind="question",
    )
    assert "missing_original" in g.node_view("question")["gaps"]
    ref = original(e, b, "one", "On November 1, 2026, Aurora announced the decision.")
    record(g, "question", finding(ref), revision=1, kind="question")
    assert g.node_view("question")["gaps"] == []
    assert len(g.data["nodes"]["question"]["findings"]) == 2
    other = finding(ref).model_copy(update={"stance": "contradicts"})
    record(g, "question", other, revision=2, kind="question")
    assert g.node_view("question")["gaps"] == ["conflicting_evidence"]


def test_plan_alone_cannot_establish_user_premise(tmp_path):
    _, e, b, g = setup(tmp_path)
    scope = Scope(
        direction="both",
        anchor="A claim supplied by the user",
        objective="Verify it",
        premise="source_supported",
    )
    with pytest.raises(ValueError, match="premise node"):
        g.set_scope(scope, 0)
    ref = original(e, b, "one", "On November 1, 2026, Aurora announced the decision.")
    record(g, "premise", finding(ref), kind="premise")
    g.set_scope(scope, 0)
    assert g.summary()["scope"]["premise"] == "source_supported"


def test_negated_causal_statement_is_not_a_positive_cause(tmp_path):
    _, e, b, g = setup(tmp_path)
    ref = original(
        e,
        b,
        "one",
        "On November 1, 2026, Aurora said there is no evidence the supplier caused the fault.",
    )
    record(g, "supplier", finding(ref))
    record(g, "fault", finding(ref))
    with pytest.raises(ValueError, match="positive causal"):
        g.relate("supplier", "fault", "cause_stated", [ref], "Causal words appear")


def test_earlier_node_remains_addressable_and_focused_after_many_new_nodes(tmp_path):
    _, e, b, g = setup(tmp_path)
    ref = original(e, b, "one", "On November 1, 2026, Aurora announced the decision.")
    for i in range(7):
        record(g, "event_" + str(i), finding(ref))
    assert "event_0" not in [n["node_id"] for n in g.summary()["nodes"]]
    assert "event_0" in g.summary()["node_ids"]
    inspect = next(t for t in graph_tools(g) if t.name == "inspect_event_graph")
    result = asyncio.run(inspect.ainvoke({"node_id": "event_0"}))
    assert '"node_id": "event_0"' in result
    assert g.summary()["nodes"][0]["node_id"] == "event_0"
    assert b.state["model_calls"] == 0
