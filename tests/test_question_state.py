import asyncio
import copy
import json

import pytest
from test_answer_contract import observation, task_at

from news_agent.answer_contract import AnswerBindings, build_state
from news_agent.budget import Budget
from news_agent.core_context import CoreEvidenceContext
from news_agent.evidence_chain import expose_passages
from news_agent.mock_model import FixtureChatModel
from news_agent.progress import plan_tool
from news_agent.question_state import QuestionState, digest, state_tool
from news_agent.schemas import BudgetConfig, DocumentVersion
from news_agent.storage import EvidenceStore, RunStore, read_json, write_json


def setup(root):
    task = task_at().model_copy(update={"synthetic": True})
    e = EvidenceStore(RunStore(root))
    b = Budget(BudgetConfig(), e.store)
    s = QuestionState(task, e, b)
    e.question_state = s
    s.register_plan(
        [
            {
                "step_id": "status",
                "title": "Find closing original",
                "kind": "plan",
                "answer_role": "main_status",
            }
        ]
    )
    return task, e, b, s


def deliver(e, b, s, vid="one", day="13", text=None, question="status"):
    text = text or f"On October {day}, 2023, Aurora completed its acquisition of Birch."
    p = e.add_document(
        DocumentVersion(
            version_id=vid, title="Aurora record", url=f"https://issuer.example/{vid}", content=text
        )
    )[0]
    s.link_candidates(question, [{"version_id": vid}])
    expose_passages(e, b, [p], method="ranked_read", query="completion")
    return p


def bind(s, p, day="2023-10-13", qid="main", **kwargs):
    from news_agent.answer_contract import Observation

    obj = Observation(**observation(p.text, day, passage_id=p.passage_id, **kwargs))
    return s.bind(qid, obj, s.data["questions"][s.resolve(qid)]["evidence_revision"])


def test_delivery_and_plan_completed_are_not_an_answer_but_bound_fact_updates_gap(tmp_path):
    _, e, b, s = setup(tmp_path)
    p = deliver(e, b, s)
    assert s.summary("main")[0]["status"] == "evidence_received"
    s.register_plan(
        [
            {
                "step_id": "status",
                "title": "Done",
                "kind": "plan",
                "answer_role": "main_status",
                "status": "completed",
            }
        ]
    )
    assert s.summary("main")[0]["gap_status"] == "open_unassessed"
    bind(s, p)
    q = s.summary("status")[0]
    assert q["status"] == "answered_provisionally" and q["gap_status"] == "addressed_model_assessed"
    assert b.state["model_calls"] == 0 and b.state.get("retrieval_api_calls", 0) == 0


def test_translated_subject_is_rejected_with_current_state_and_exact_source_retry(tmp_path):
    _, e, b, s = setup(tmp_path)
    p = deliver(e, b, s)
    update = state_tool(s)
    args = {
        "question_id": "main",
        "seen_revision": 1,
        "observation": observation(p.text, passage_id=p.passage_id, subject="极光公司"),
    }
    before = copy.deepcopy(s.data)
    refused = json.loads(asyncio.run(update.ainvoke(args)))
    assert refused["error"] == "subject_not_in_quote"
    assert "original language" in refused["recovery"]
    assert refused["question_state"][0]["evidence_revision"] == 1
    assert refused["question_state"][0]["gap_status"] == "open_unassessed"
    assert s.data == before
    args["observation"]["subject"] = "Aurora"
    accepted = json.loads(asyncio.run(update.ainvoke(args)))
    assert accepted["question_state"][0]["gap_status"] == "addressed_model_assessed"
    assert b.state["model_calls"] == 0 and b.state.get("retrieval_api_calls", 0) == 0


def test_new_original_reopens_review_and_stale_assessment_cannot_close_it(tmp_path):
    _, e, b, s = setup(tmp_path)
    p = deliver(e, b, s)
    bind(s, p)
    old = s.summary("main")[0]["evidence_revision"]
    deliver(e, b, s, vid="two")
    assert s.summary("main")[0]["status"] == "pending_review"
    from news_agent.answer_contract import Observation

    before = copy.deepcopy(s.data)
    with pytest.raises(ValueError, match="stale"):
        s.bind("main", Observation(**observation(p.text, passage_id=p.passage_id)), old)
    assert s.data == before
    bind(s, p)  # explicit assessment of the current revision; no claim of independent truth
    assert s.summary("main")[0]["status"] == "answered_provisionally"


def test_conflict_stays_open_under_repeat_support_plan_complete_and_final_report(tmp_path):
    task, e, b, s = setup(tmp_path)
    a = deliver(e, b, s)
    bind(s, a)
    z = deliver(e, b, s, vid="different", day="14")
    bind(s, z, day="2023-10-14")
    assert s.summary("main")[0]["status"] == "conflicted"
    bind(s, a)
    assert s.summary("main")[0]["gap_status"] == "open_conflict"
    packet = CoreEvidenceContext(FixtureChatModel(), task, e, b).packet([])
    assert a.passage_id in {pid for d in packet["documents"] for pid in d["excerpts"]}
    assert z.passage_id in {pid for d in packet["documents"] for pid in d["excerpts"]}
    result = build_state(
        AnswerBindings(observations=[observation(a.text, passage_id=a.passage_id)]), task, packet
    )
    assert result["inference"]["status"] == "unknown" and not result["inference"]["passage_ids"]


def test_approval_and_future_plan_cannot_satisfy_completion_need(tmp_path):
    _, e, b, s = setup(tmp_path)
    p = deliver(e, b, s, text="On October 13, 2023, the CMA gave Aurora consent to acquire Birch.")
    bind(s, p, kind="approval", target_event=False)
    assert s.summary("main")[0]["status"] == "partial_support"
    p = deliver(
        e,
        b,
        s,
        vid="plan",
        text="On October 13, 2023, Aurora expects the acquisition to be completed.",
    )
    bind(s, p, occurrence="planned")
    assert s.summary("main")[0]["gap_status"] == "open_requirements_missing"


def test_independent_questions_only_update_for_their_own_evidence(tmp_path):
    _, e, b, s = setup(tmp_path)
    s.register_plan(
        [
            {
                "step_id": "cma",
                "title": "Dated consultation",
                "kind": "temporal",
                "answer_role": "date_anchor",
                "evidence_kind": "consultation",
                "requires_date": True,
            }
        ]
    )
    p = deliver(
        e,
        b,
        s,
        question="cma",
        text="On September 22, 2023, the CMA is consulting on the Aurora acquisition.",
    )
    bind(
        s,
        p,
        day="2023-09-22",
        qid="cma",
        kind="consultation",
        subject="the CMA",
        target_event=False,
    )
    assert s.summary("cma")[0]["status"] == "answered_provisionally"
    assert s.summary("main")[0]["status"] == "awaiting_evidence"


@pytest.mark.parametrize("occurrence", ["planned", "unknown"])
def test_non_observed_auxiliary_approval_cannot_close_an_event_requirement(tmp_path, occurrence):
    _, e, b, s = setup(tmp_path)
    s.register_plan(
        [
            {
                "step_id": "reg",
                "title": "Approval date",
                "kind": "source_check",
                "evidence_kind": "approval",
                "requires_date": True,
            }
        ]
    )
    p = deliver(
        e,
        b,
        s,
        question="reg",
        text="On October 13, 2023, the CMA expects to approve the Aurora acquisition.",
    )
    bind(
        s,
        p,
        qid="reg",
        kind="approval",
        subject="the CMA",
        target_event=False,
        occurrence=occurrence,
    )
    assert s.summary("reg")[0]["status"] == "partial_support"
    assert s.summary("reg")[0]["gap_status"] == "open_requirements_missing"


def test_unread_or_wrong_quote_and_question_redefinition_leave_state_unchanged(tmp_path):
    _, e, b, s = setup(tmp_path)
    p = deliver(e, b, s)
    from news_agent.answer_contract import Observation

    before = copy.deepcopy(s.data)
    for obj in [
        observation("A fabricated source quotation", passage_id=p.passage_id),
        observation(p.text, passage_id="not-seen"),
    ]:
        with pytest.raises(ValueError):
            s.bind("main", Observation(**obj), 1)
        assert s.data == before
    s.register_plan(
        [
            {
                "step_id": "reg",
                "title": "Approval",
                "kind": "source_check",
                "evidence_kind": "approval",
                "requires_date": True,
            }
        ]
    )
    before = copy.deepcopy(s.data)
    with pytest.raises(ValueError, match="immutable"):
        s.register_plan(
            [
                {
                    "step_id": "reg",
                    "title": "Changed",
                    "kind": "source_check",
                    "evidence_kind": "completion",
                    "requires_date": True,
                }
            ]
        )
    assert s.data == before
    with pytest.raises(ValueError, match="alias"):
        s.register_plan(
            [
                {
                    "step_id": "status",
                    "title": "Different question",
                    "kind": "temporal",
                    "evidence_kind": "consultation",
                    "requires_date": True,
                }
            ]
        )
    assert s.data == before


def test_resume_dedup_and_unavailability_do_not_certify_facts(tmp_path):
    task, e, b, s = setup(tmp_path)
    p = deliver(e, b, s)
    bind(s, p)
    before = (tmp_path / "trajectory.jsonl").read_text()
    bind(s, p)
    assert (tmp_path / "trajectory.jsonl").read_text() == before
    resumed = QuestionState(task, e, b)
    assert resumed.data == s.data
    resumed.finish("research_budget_exhausted")
    assert resumed.summary("main")[0]["status"] == "answered_provisionally"
    data = read_json(tmp_path / "question_state.json")
    data["questions"]["main"]["status"] = "verified"
    write_json(tmp_path / "question_state.json", data)
    with pytest.raises(ValueError, match="hash mismatch"):
        QuestionState(task, e, b)


def test_stop_without_evidence_keeps_gap_open(tmp_path):
    _, _, _, s = setup(tmp_path)
    s.unavailable("main", "source could not be obtained")
    assert s.summary("main")[0]["gap_status"] == "open_unavailable"
    s.finish("deadline_reached")
    assert s.summary("main")[0]["status"] == "unavailable"


def test_old_policy_resume_is_rejected_without_rewriting_historical_state(tmp_path):
    task, e, b, s = setup(tmp_path)
    data = copy.deepcopy(s.data)
    data["version"] = "question-evidence-state-v1"
    data["sha256"] = digest({k: v for k, v in data.items() if k != "sha256"})
    write_json(tmp_path / "question_state.json", data)
    before = (tmp_path / "question_state.json").read_bytes()
    journal = (tmp_path / "trajectory.jsonl").read_bytes()
    with pytest.raises(ValueError, match="policy/task changed"):
        QuestionState(task, e, b)
    assert (tmp_path / "question_state.json").read_bytes() == before
    assert (tmp_path / "trajectory.jsonl").read_bytes() == journal


def test_plan_partial_updates_preserve_the_registered_criterion(tmp_path):
    _, e, b, s = setup(tmp_path)
    publish = plan_tool(e, b)
    first = {
        "step_id": "reg",
        "title": "CMA date",
        "kind": "temporal",
        "answer_role": "date_anchor",
        "evidence_kind": "consultation",
        "requires_date": True,
        "status": "in_progress",
    }
    assert "error" not in json.loads(asyncio.run(publish.ainvoke({"steps": [first]})))
    result = json.loads(
        asyncio.run(
            publish.ainvoke(
                {"steps": [{"step_id": "reg", "title": "CMA date", "status": "completed"}]}
            )
        )
    )
    assert "error" not in result
    assert s.data["questions"]["reg"]["kind"] == "consultation"
    assert s.data["questions"]["reg"]["status"] == "awaiting_evidence"


def test_final_revision_cache_remains_valid_after_its_own_state_update(tmp_path):
    from test_revision import RevisionModel

    from news_agent.budget import BudgetCallback
    from news_agent.revision import revise_report

    task, e, b, s = setup(tmp_path)
    p = deliver(e, b, s)
    model = RevisionModel(
        reply=json.dumps({"observations": [observation(p.text, passage_id=p.passage_id)]})
    )
    context = CoreEvidenceContext(model, task, e, b)
    config = {"callbacks": [BudgetCallback(b, 0)]}
    answer, record = asyncio.run(revise_report("Draft", task, context, None, model, b, config))
    assert (
        record["status"] == "applied" and s.summary("main")[0]["status"] == "answered_provisionally"
    )
    again, _ = asyncio.run(revise_report("Draft", task, context, None, model, b, config))
    assert again == answer and model.requests == 1
    deliver(e, b, s, vid="new", day="14")
    changed, newrecord = asyncio.run(revise_report("Draft", task, context, None, model, b, config))
    assert newrecord["status"] == "safe_partial" and "无法确定" in changed and model.requests == 1


def test_committed_transition_missing_from_journal_is_recovered_without_rebinding(tmp_path):
    task, e, b, s = setup(tmp_path)
    p = deliver(e, b, s)
    bind(s, p)
    journal = tmp_path / "trajectory.jsonl"
    lines = journal.read_text().splitlines()
    assert json.loads(lines[-1])["kind"] == "question_state_changed"
    journal.write_text(
        "\n".join(lines[:-1]) + "\n"
    )  # crash after checkpoint, before event publication
    reopened_e = EvidenceStore(RunStore(tmp_path))
    reopened_b = Budget(BudgetConfig(), reopened_e.store)
    resumed = QuestionState(task, reopened_e, reopened_b)
    assert resumed.data == s.data and reopened_b.state["model_calls"] == 0
    event = json.loads(journal.read_text().splitlines()[-1])
    assert event["replayed_after_checkpoint"] and event["after_status"] == "answered_provisionally"
    after = journal.read_text()
    QuestionState(task, EvidenceStore(RunStore(tmp_path)), reopened_b)
    assert journal.read_text() == after
