import copy
import time

import pytest
from test_question_state import bind, deliver, setup

from news_agent.repair_admission import RepairAdmission
from news_agent.storage import read_json, write_json


def partial(root):
    task, e, b, s = setup(root)
    p = deliver(e, b, s, text="On October 13, 2023, the CMA gave Aurora consent to acquire Birch.")
    bind(s, p, kind="approval", target_event=False)
    assert s.summary("main")[0]["status"] == "partial_support"
    return task, e, b, s


def test_recoverable_gap_gets_one_persistent_slot_without_budget_or_network_usage(tmp_path):
    _, _, budget, state = partial(tmp_path)
    before = copy.deepcopy(budget.state)
    gate = RepairAdmission(state, budget)
    assert not (tmp_path / "repair_state.json").exists()
    decision = gate.check(5000)
    assert decision["admitted"] and decision["minimum_tokens"] == 19500
    stale_handle = RepairAdmission(state, budget)
    attempt = gate.reserve(5000)
    assert attempt["status"] == "reserved_not_executed" and not attempt["automatic_execution"]
    assert gate.validate_for_dispatch()["admitted"]
    assert stale_handle.reserve(5000)["reason"] == "repair_attempt_already_consumed"
    resumed = RepairAdmission(state, budget)
    assert resumed.reserve(5000)["reason"] == "repair_attempt_already_consumed"
    resumed.cancel("source unavailable; no dispatch")
    assert resumed.validate_for_dispatch()["reason"] == "no_reserved_attempt"
    assert (
        RepairAdmission(state, budget).reserve(5000)["reason"] == "repair_attempt_already_consumed"
    )
    assert budget.state == before
    assert read_json(tmp_path / "budget.json") == before
    assert read_json(tmp_path / "repair_state.json")["attempt"]["status"] == (
        "cancelled_without_execution"
    )


@pytest.mark.parametrize(
    "changed,reason",
    [
        ({"phase": "writing"}, "research_closed_preserve_writing_reserve"),
        ({"phase": "repair"}, "research_closed_preserve_writing_reserve"),
        ({"model_calls": 16}, "insufficient_research_model_calls"),
        ({"tool_calls": 22}, "insufficient_research_tool_calls"),
        ({"tokens": 41000}, "insufficient_research_tokens"),
        ({"search_rounds": 6}, "search_round_limit"),
        ({"pending": {"unknown-request": 40000}}, "request_in_flight"),
        ({"retrieval_pending": {"search-1": "search"}}, "request_in_flight"),
        ({"deadline_at": 0}, "insufficient_research_time"),
    ],
)
def test_admission_keeps_original_limits_and_never_reopens_research(tmp_path, changed, reason):
    _, _, b, s = partial(tmp_path)
    b.state.update(changed)
    before = copy.deepcopy(b.state)
    gate = RepairAdmission(s, b)
    assert gate.reserve(5000)["reason"] == reason
    assert b.state == before and not (tmp_path / "repair_state.json").exists()


def test_new_evidence_or_spent_capacity_invalidates_reserved_slot_without_refund(tmp_path):
    _, e, b, s = partial(tmp_path)
    gate = RepairAdmission(s, b)
    gate.reserve(5000)
    b.state["deadline_at"] = time.time() + 200
    assert gate.validate_for_dispatch()["reason"] == "insufficient_research_time"
    b.state["deadline_at"] = time.time() + 600
    p = deliver(e, b, s, vid="late")
    bind(s, p)
    assert gate.validate_for_dispatch()["reason"] == "evidence_or_question_state_changed"
    assert gate.reserve(5000)["reason"] == "repair_attempt_already_consumed"


def test_existing_material_gets_assessment_and_conflict_never_automatic_arbitration(tmp_path):
    _, e, b, s = setup(tmp_path)
    gate = RepairAdmission(s, b)
    assert gate.check(5000)["action"] == "initial_research"
    a = deliver(e, b, s)
    assert gate.check(5000)["action"] == "assess_saved"
    bind(s, a)
    assert gate.check(5000)["reason"] == "already_provisionally_covered"
    other = deliver(e, b, s, vid="other", day="14")
    assert gate.check(5000)["action"] == "assess_saved"
    bind(s, other, day="2023-10-14")
    assert gate.check(5000)["reason"] == "conflict_requires_explicit_review"
    assert not (tmp_path / "repair_state.json").exists()


def test_unavailable_stopped_undefined_or_auxiliary_gaps_do_not_auto_search(tmp_path):
    _, _, b, s = partial(tmp_path)
    gate = RepairAdmission(s, b)
    s.unavailable("main", "official source not accessible")
    assert gate.check(5000)["reason"] == "source_unavailability_not_an_automatic_retry"
    s.finish("deadline_reached")
    assert gate.check(5000)["reason"] == "run_already_stopped"
    s.register_plan([{"step_id": "other", "title": "Unknown need", "kind": "gap"}])
    assert gate.check(5000, "other")["reason"] == "outside_typed_main_completion_scope"


def test_missing_cutoff_is_input_scope_problem_and_tampering_cannot_reset_slot(tmp_path):
    _, _, b, s = partial(tmp_path)
    s.task = s.task.model_copy(
        update={"question": "Was the acquisition completed?", "cutoff": None}
    )
    gate = RepairAdmission(s, b)
    assert gate.check(5000)["reason"] == "missing_cutoff_requires_user_scope_not_search"
    s.task = s.task.model_copy(
        update={"question": "As of 2023-09-30 was the acquisition completed?"}
    )
    gate.reserve(5000)
    saved = read_json(tmp_path / "repair_state.json")
    saved["attempt"] = None
    write_json(tmp_path / "repair_state.json", saved)
    with pytest.raises(ValueError, match="checksum"):
        RepairAdmission(s, b)


@pytest.mark.parametrize("estimate", [0, -1, True, 0.5, "5000"])
def test_request_estimate_must_be_explicit_positive_integer(tmp_path, estimate):
    _, _, b, s = partial(tmp_path)
    with pytest.raises(ValueError, match="estimate"):
        RepairAdmission(s, b).check(estimate)
