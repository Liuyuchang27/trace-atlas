import asyncio
import json
from datetime import datetime

import pytest
from test_revision import RevisionModel

from news_agent.answer_contract import AnswerBindings, build_state, cutoff_for, parse_and_render
from news_agent.budget import Budget, BudgetCallback
from news_agent.core_context import CoreEvidenceContext
from news_agent.evidence_chain import core_policy, expose_passages
from news_agent.revision import finalize_unbound, revise_report
from news_agent.schemas import BudgetConfig, DocumentVersion, ResearchTask
from news_agent.storage import EvidenceStore, RunStore, read_json


def task_at(day="2023-09-30"):
    return ResearchTask(
        task_id="binding-test",
        event="Aurora acquisition of Birch",
        question=f"截至{day}，Aurora收购Birch是否已完成？",
        mode="retrospective",
    )


def packet(text, pid="p-1111111111111111"):
    return {
        "packet_id": "fixture",
        "documents": [{"version_id": "issuer", "excerpts": {pid: text}}],
    }


def observation(text, day="2023-10-13", **kw):
    return {
        "kind": "completion",
        "subject": "Aurora",
        "target_event": True,
        "occurrence": "observed",
        "event_date": day,
        "passage_id": "p-1111111111111111",
        "quote": text,
        **kw,
    }


def state(text, day="2023-10-13", task=None, **kw):
    return build_state(
        AnswerBindings(observations=[observation(text, day, **kw)]), task or task_at(), packet(text)
    )


@pytest.mark.parametrize(
    "cutoff,status",
    [
        ("2023-09-30", "not_completed_by_cutoff"),
        ("2023-10-13", "completed_by_cutoff"),
        ("2023-10-31", "completed_by_cutoff"),
    ],
)
def test_cutoff_answer_and_timeline_share_dated_completion_original(cutoff, status):
    text = "On Oct. 13, 2023, Aurora completed its acquisition of Birch."
    report, result = parse_and_render(
        json.dumps({"observations": [observation(text)]}), task_at(cutoff), packet(text)
    )
    assert result["inference"]["status"] == status
    assert result["inference"]["passage_ids"] == ["p-1111111111111111"]
    assert "据完成日期回溯推断[p-1111111111111111]" in report
    assert "监管批准或案件关闭不能单独替代" in report
    assert text in report


@pytest.mark.parametrize(
    "text",
    [
        "On October 13, 2023, Aurora expected the acquisition to be completed.",
        "On October 13, 2023, Aurora had not completed the acquisition of Birch.",
        "On October 13, 2023, the CMA case closed on the Aurora acquisition.",
        "On October 13, 2023, Aurora completed its review of the Birch acquisition.",
        "On October 13, 2023, the CMA gave Aurora consent to acquire Birch.",
    ],
)
def test_plans_negative_permission_and_procedure_cannot_be_promoted_to_closing(text):
    result = state(text)
    assert result["inference"]["status"] == "unknown" and result["rejected"]


def test_negation_cannot_be_removed_by_selecting_a_smaller_exact_quote():
    original = "On October 13, 2023, Aurora had not completed the acquisition of Birch."
    selected = "completed the acquisition of Birch."
    result = build_state(
        AnswerBindings(observations=[observation(selected, None, subject="Birch")]),
        task_at(),
        packet(original),
    )
    assert result["rejected"][0]["reason"] == "not_an_observed_completion_statement"


def test_planned_future_closing_and_earlier_approval_leave_status_unknown():
    text = "On October 13, 2023, Aurora expects the acquisition to be completed."
    result = state(text, occurrence="planned")
    assert result["inference"]["status"] == "unknown" and not result["rejected"]
    text = "On September 22, 2023, the CMA gave Aurora consent to acquire Birch."
    result = state(text, day="2023-09-22", kind="approval")
    assert result["inference"]["status"] == "unknown"


def test_unseen_original_wrong_date_and_conflicting_closing_dates_are_not_conclusions():
    text = "On October 13, 2023, Aurora completed its acquisition of Birch."
    result = state(text, day="2023-09-22")
    assert result["inference"]["status"] == "unknown" and result["rejected"]
    wrong = observation(text, passage_id="p-not-shown")
    result = build_state(AnswerBindings(observations=[wrong]), task_at(), packet(text))
    assert result["rejected"][0]["reason"] == "quote_not_in_actual_input"
    later = text.replace("13,", "14,")
    p = packet(text)
    p["documents"][0]["excerpts"]["p-2222222222222222"] = later
    result = build_state(
        AnswerBindings(
            observations=[
                observation(text),
                observation(later, "2023-10-14", passage_id="p-2222222222222222"),
            ]
        ),
        task_at(),
        p,
    )
    assert result["inference"]["rule"] == "conflicting_completion_dates"


def test_nearest_event_date_is_not_the_earlier_announcement_date():
    text = (
        "On January 18, 2022 Aurora announced it had completed the acquisition on October 13, 2023."
    )
    result = state(text, day="2022-01-18")
    assert result["rejected"][0]["reason"] == "date_not_nearest_completion_predicate"


def test_cutoff_is_from_user_task_not_document_clock_or_another_question_date():
    task = task_at().model_copy(
        update={
            "question": (
                "截至2023年9月30日，收购是否完成？请区分2023年10月13日的消息。"
                '\n\nRun observation context: {"utc":"2026-11-13"}'
            )
        }
    )
    assert cutoff_for(task)["date"] == "2023-09-30"
    assert (
        cutoff_for(
            task.model_copy(update={"question": "截至2023-09-30及截至2023-10-31分别怎样？"})
        )["date"]
        is None
    )
    exact = task.model_copy(update={"cutoff": datetime.fromisoformat("2023-10-13T12:00:00+00:00")})
    assert (
        state("On October 13, 2023, Aurora completed its acquisition of Birch.", task=exact)[
            "inference"
        ]["rule"]
        == "same_day_time_unknown"
    )


def test_commitment_recipient_direction_is_literal_and_permission_is_not_completion():
    text = (
        "Aurora made commitments to the European Commission "
        "and UK Competition and Markets Authority."
    )
    row = {
        "kind": "commitments_to",
        "passage_id": "p-1111111111111111",
        "quote": text,
        "recipient": "the European Commission and UK Competition and Markets Authority",
    }
    report, result = parse_and_render(
        json.dumps({"legal_relations": [row]}), task_at(), packet(text)
    )
    assert "表示向the European Commission" in report
    assert "获得欧盟" not in report and not result["rejected"]
    row["recipient"] = "Aurora"
    result = build_state(AnswerBindings(legal_relations=[row]), task_at(), packet(text))
    assert result["rejected"][0]["reason"] == "commitment_recipient_not_after_to"


def test_publication_date_is_not_a_completion_date():
    text = "Published: October 13, 2023\nAurora completed its acquisition of Birch."
    result = state(text)
    assert result["inference"]["status"] == "unknown" and result["rejected"]


def test_permission_and_in_lieu_terms_preserve_their_legal_stage():
    text = "The CMA gave consent to acquire Birch and accepted undertakings in lieu of a reference."
    terms = [
        {"kind": kind, "passage_id": "p-1111111111111111", "quote": text}
        for kind in ["consent_to_acquire", "undertakings_in_lieu"]
    ]
    report, result = parse_and_render(
        json.dumps({"legal_relations": terms}), task_at(), packet(text)
    )
    assert not result["rejected"] and result["inference"]["status"] == "unknown"
    assert "监管许可" in report and "以承诺替代将案件移交" in report


@pytest.mark.parametrize(
    "reply,finish", [(None, "stop"), ("not-json", "stop"), ("truncated", "length")]
)
def test_structured_revision_has_one_call_and_safe_unknown_on_failure(tmp_path, reply, finish):
    task = task_at()
    store = RunStore(tmp_path)
    e = EvidenceStore(store)
    b = Budget(BudgetConfig(), store)
    b.state["evidence_chain_version"] = core_policy()["version"]
    text = "On October 13, 2023, Aurora completed its acquisition of Birch."
    passages = e.add_document(
        DocumentVersion(
            version_id="issuer",
            title="Issuer record",
            url="https://issuer.example/record",
            content=text,
        )
    )
    expose_passages(e, b, passages, method="ranked_read", query="completed acquisition")
    context = CoreEvidenceContext(RevisionModel(reply=""), task, e, b)
    if reply is None:
        reply = json.dumps({"observations": [observation(text, passage_id=passages[0].passage_id)]})
    model = RevisionModel(reply=reply, finish=finish)
    config = {"callbacks": [BudgetCallback(b, 0)]}
    draft = "Wrong draft: approval proves closing; regulators made commitments."
    output, record = asyncio.run(revise_report(draft, task, context, None, model, b, config))
    assert record["status"] == (
        "applied" if finish == "stop" and reply.startswith("{") else "safe_partial"
    )
    assert model.requests == 1 and "Wrong draft" not in output
    assert (tmp_path / "report_draft.md").read_text() == draft
    assert (tmp_path / "report_revision_raw_response.txt").read_text() == reply
    saved = read_json(tmp_path / "answer_state.json")
    assert saved["inference"]["status"] == (
        "not_completed_by_cutoff" if record["status"] == "applied" else "unknown"
    )
    repeated, _ = asyncio.run(revise_report(draft, task, context, None, model, b, config))
    assert model.requests == 1
    if record["status"] == "applied":
        assert repeated == output


def test_core_budget_refusal_does_not_promote_the_old_draft(tmp_path):
    task = task_at()
    store = RunStore(tmp_path)
    e = EvidenceStore(store)
    b = Budget(BudgetConfig(), store)
    text = "On October 13, 2023, Aurora completed its acquisition of Birch."
    p = e.add_document(
        DocumentVersion(
            version_id="issuer", title="Issuer", url="https://issuer.example/", content=text
        )
    )[0]
    expose_passages(e, b, [p], method="ranked_read", query="completion")
    b.state["tokens"] = 79900
    model = RevisionModel(reply="{}")
    context = CoreEvidenceContext(model, task, e, b)
    report, record = asyncio.run(
        revise_report(
            "Unsupported certainty",
            task,
            context,
            None,
            model,
            b,
            {"callbacks": [BudgetCallback(b, 0)]},
        )
    )
    assert model.requests == 0 and record["status"] == "safe_partial"
    assert "无法确定" in report and "Unsupported certainty" not in report


def test_bad_auxiliary_item_does_not_discard_valid_completion_but_is_recorded():
    text = "On October 13, 2023, Aurora completed its acquisition of Birch."
    malformed = observation(text, kind="consultation", subject="Too long " * 30, target_event=False)
    report, result = parse_and_render(
        json.dumps({"observations": [observation(text), malformed]}), task_at(), packet(text)
    )
    assert result["inference"]["status"] == "not_completed_by_cutoff"
    assert (
        len(result["observations"]) == 1
        and result["rejected"][0]["reason"] == "observations_schema"
    )
    assert "未通过" in report


def test_missing_final_model_output_never_displays_bad_claim_paraphrases(tmp_path):
    task = task_at()
    store = RunStore(tmp_path)
    e = EvidenceStore(store)
    b = Budget(BudgetConfig(), store)
    text = "Aurora made commitments to the European Commission."
    p = e.add_document(
        DocumentVersion(
            version_id="source", title="Record", url="https://issuer.example/", content=text
        )
    )[0]
    e.add_claim("获得监管机构承诺后已完成", [p.passage_id], "2023-10-13", "completion")
    model = RevisionModel(reply="must not call")
    context = CoreEvidenceContext(model, task, e, b)
    report, record = finalize_unbound(task, context, "deadline_reached")
    assert "无法确定" in report and "获得监管机构" not in report
    assert record["model_calls"] == model.requests == 0
    assert "截止状态：无法确定" in report
    assert e.claims[0].text == "获得监管机构承诺后已完成"  # raw trace retained, not promoted


def test_sdk_timeout_preserves_raw_claims_but_uses_safe_report_and_timeline(
    tmp_path, task, corpus_path, monkeypatch
):
    import news_agent.harness as harness
    from news_agent.mock_model import FixtureChatModel
    from news_agent.web_view import build_snapshot

    class StopAfterClaim(FixtureChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            for m in messages:
                if m.type == "human" and m.text.startswith("Here is a summary"):
                    packet = json.loads(m.text.split("\n\n", 1)[1])
                    if packet.get("claims"):
                        raise TimeoutError("mock_deadline_after_claim")
            result = super()._generate(messages, stop, run_manager, **kwargs)
            for call in result.generations[0].message.tool_calls:
                if call["name"] == "record_claim":
                    call["args"]["text"] = "获得监管机构承诺后已完成"
            return result

    monkeypatch.setattr(harness, "FixtureChatModel", StopAfterClaim)
    result = asyncio.run(
        harness.run_research(
            tmp_path, task, corpus_path, BudgetConfig(), mock=True, visual_progress=True
        )
    )
    assert result["status"] == "partial" and not result["model_finalized"]
    assert result["report_source"] == "safe_binding_fallback"
    assert result["budget"]["unknown_usage_calls"] == 1 and not result["budget"]["pending"]
    assert "获得监管机构" in (tmp_path / "report_unvalidated_claims.md").read_text()
    assert "获得监管机构" not in (tmp_path / "report.md").read_text()
    snapshot = build_snapshot(tmp_path)
    assert snapshot["timeline"] and all(
        "获得监管机构" not in row["text"] for row in snapshot["timeline"]
    )


def test_sdk_provider_failure_uses_only_original_revision_and_preserves_unknown_usage(
    tmp_path, monkeypatch
):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    import news_agent.harness as harness
    from news_agent.mock_model import FixtureChatModel

    class APIConnectionError(RuntimeError):
        pass

    class RecoverFinalizer(FixtureChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if messages[0].text.startswith("REPORT_REVISION:"):
                p = json.loads(messages[-1].text)["eligible_evidence"]
                excerpts = p["documents"][0]["excerpts"]
                pid, text = next((pid, t) for pid, t in excerpts.items() if "completed" in t)
                content = json.dumps({"observations": [observation(text, passage_id=pid)]})
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=content,
                                response_metadata={"finish_reason": "stop"},
                                usage_metadata={
                                    "input_tokens": 100,
                                    "output_tokens": 50,
                                    "total_tokens": 150,
                                },
                            )
                        )
                    ]
                )
            for m in messages:
                if m.type == "human" and m.text.startswith("Here is a summary"):
                    if json.loads(m.text.split("\n\n", 1)[1]).get("claims"):
                        raise APIConnectionError("simulated uncertain request")
            return super()._generate(messages, stop, run_manager, **kwargs)

    doc = DocumentVersion(
        version_id="issuer",
        title="Aurora Birch acquisition completion",
        url="https://issuer.example/record",
        content="On October 13, 2023, Aurora completed its acquisition of Birch.",
    )
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([doc.model_dump(mode="json")]))
    monkeypatch.setattr(harness, "FixtureChatModel", RecoverFinalizer)
    result = asyncio.run(
        harness.run_research(
            tmp_path / "run", task_at(), corpus, BudgetConfig(), mock=True, visual_progress=True
        )
    )
    assert result["status"] == "partial" and result["stop_reason"] == "APIConnectionError"
    assert result["budget"]["unknown_usage_calls"] == 1 and not result["budget"]["pending"]
    assert (
        result["report_revision"]["model_calls"] == 1
        and result["report_source"] == "model_bindings_template"
    )
    s = read_json(tmp_path / "run/answer_state.json")
    assert s["inference"]["status"] == "not_completed_by_cutoff"
    assert result["budget"]["tokens"] <= 80000
