"""Projection and web workflow checks use synthetic data and no external calls."""

import asyncio
import json

from question_parts_fixture import bind, prepared
from test_event_graph import original

from news_agent.event_graph import Finding, TimeSpan
from news_agent.event_report import evidence_dossier
from news_agent.schemas import ResearchTask
from news_agent.storage import read_json
from news_agent.temporal_review import digest
from news_agent.web_view import build_snapshot, report_html
from news_agent.web_worker import work


def test_unknown_finding_visible_and_report_dossier_quotes_same_original(tmp_path):
    _, evidence, budget, graph = prepared(tmp_path)
    ref = original(evidence, budget, "launch", "5月3日发射成功。")
    bind(graph, ref, "launch", "observed", "原文记载5月3日发射，年份未知。")
    view = build_snapshot(tmp_path)
    assert view["timeline"][0]["date"] == "日期未知"
    assert view["timeline"][0]["citations"][0]["passage_id"] == ref.passage_id
    dossier = "\n".join(evidence_dossier(evidence))
    assert ref.quote in dossier and "问题覆盖" in dossier
    assert graph.data["nodes"]["q"]["findings"][-1]["finding_id"] in dossier
    assert "入轨是否完成？" in dossier and "尚未满足" in dossier
    view["report"] = dossier + "\n<script>alert(1)</script>"
    exported = report_html(view)
    assert "<script>alert(1)</script>" not in exported
    fid = graph.data["nodes"]["q"]["findings"][-1]["finding_id"]
    assert f'href="#{fid}"' in exported and f'id="{fid}"' in exported
    # The export's timeline remains navigable even if the short report omits this evidence.
    view["report"] = "Partial report; see timeline for saved findings."
    assert f'id="{ref.passage_id}"' in report_html(view)


def test_report_brief_preserves_prior_review_and_hides_duplicate_background(tmp_path):
    _, evidence, budget, graph = prepared(tmp_path)
    ref = original(evidence, budget, "launch", "探测器于2026年5月3日发射成功。")
    graph.append(
        "q",
        Finding(
            assertion="发射已经完成。",
            stance="supports",
            occurrence="observed",
            event_time=TimeSpan(precision="day", start="2026-05-03"),
            originals=[ref],
            answers_question_part=True,
            question_part_id="launch",
        ),
        1,
    )
    background = original(evidence, budget, "background", "报道说发射在5月。")
    bind(graph, background, "launch", "observed", "报道提到5月。")
    fid = graph.data["nodes"]["q"]["findings"][1]["finding_id"]
    approved = {
        "status": "reviewed",
        "graph_sha256": graph.data["sha256"],
        "decisions": [
            {
                "finding_id": fid,
                "event_verdict": "supported",
                "applicability": "not_applicable",
                "event_reason": "原文明确给出年月日。",
                "applicability_reason": "一次性事件。",
            },
        ],
    }
    approved["sha256"] = digest(approved)
    review_path = "temporal_reviews/prior/record.json"
    evidence.store.save(review_path, approved)
    evidence.store.save("effective_config.json", {"workflow": "adaptive-event-map"})
    evidence.store.save("temporal_review.json", {"status": "not_reviewed"})
    (tmp_path / "report.md").write_text("Deprecated markdown with duplicated audit logs")

    view = build_snapshot(tmp_path)
    brief = view["report_brief"]
    assert brief["repeat_review_incomplete"]
    assert [a["question"] for a in brief["answers"]] == ["发射是否完成？"]
    assert brief["timeline"] == [{"date": "2026-05-03", "ids": [fid]}]
    assert brief["background"][0][0]["date"] is None
    exported = report_html(view)
    assert exported.count("发射已经完成。") == 1
    assert "Deprecated markdown" not in exported
    assert f'href="#{fid}"' in exported and f'id="{ref.passage_id}"' in exported

    # A prior decision for a different graph must not be reused for the visible record.
    approved["graph_sha256"] = "different"
    approved["sha256"] = digest({k: v for k, v in approved.items() if k != "sha256"})
    evidence.store.save(review_path, approved)
    view = build_snapshot(tmp_path)
    assert not view["report_brief"]["has_review"]
    assert view["timeline"][0]["temporal_verdict"] == "not_reviewed"


def test_web_worker_passes_explicit_adaptive_workflow_without_real_api(tmp_path, monkeypatch):
    import news_agent.web_worker as worker

    (tmp_path / "data/fixtures").mkdir(parents=True)
    task = ResearchTask(
        task_id="demo", event="demo", question="demo", mode="retrospective", synthetic=True
    )
    (tmp_path / "data/fixtures/task.json").write_text(task.model_dump_json())
    run = tmp_path / "run"
    run.mkdir()
    (run / "web_request.json").write_text(
        json.dumps({"mode": "fixture", "condition": "baseline", "workflow": "adaptive-event-map"})
    )
    observed = []

    async def fake_run(*args, **kwargs):
        observed.append(kwargs)
        return {"status": "partial"}

    monkeypatch.setattr(worker, "run_research", fake_run)
    asyncio.run(work(tmp_path, run))
    assert observed[0]["workflow"] == "adaptive-event-map"
    assert observed[0]["mock"] is True
    assert read_json(run / "worker_status.json")["status"] == "partial"
