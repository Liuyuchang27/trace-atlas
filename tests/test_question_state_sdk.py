import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_answer_contract import observation, task_at

import news_agent.harness as harness
from news_agent.mock_model import FixtureChatModel
from news_agent.schemas import BudgetConfig, DocumentVersion
from news_agent.storage import read_json
from news_agent.web_view import build_snapshot


class ConflictScenario(FixtureChatModel):
    """Scripted SDK transition acceptance; no research/semantic quality evaluation."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        packet = None
        for m in messages:
            if m.type == "human" and m.text.startswith("Here is a summary"):
                packet = json.loads(m.text.split("\n\n", 1)[1])
            elif m.type == "human" and m.text.startswith("C3 bounded evidence step."):
                packet = json.loads(m.text.split("\n", 1)[1])
        if messages[0].text.startswith("REPORT_REVISION:"):
            p = json.loads(messages[-1].text)["eligible_evidence"]
            doc = next(d for d in p["documents"] if d["version_id"] == "first")
            pid, text = next(iter(doc["excerpts"].items()))
            content = json.dumps({"observations": [observation(text, passage_id=pid)]})
            calls = []  # deliberately omit the second closing; server must preserve the conflict
        else:
            n = self.calls
            content = ""
            calls = []
            if n == 1:
                actions = [
                    (
                        "publish_plan",
                        {
                            "steps": [
                                {
                                    "step_id": "status",
                                    "title": "Find closing date",
                                    "answer_role": "main_status",
                                    "status": "in_progress",
                                }
                            ]
                        },
                    ),
                    (
                        "search",
                        {
                            "query": "First Aurora Birch",
                            "purpose": "Find first original",
                            "intent": "initial",
                            "step_id": "main",
                            "limit": 1,
                        },
                    ),
                ]
            elif n == 2:
                actions = [("read_document", {"version_id": "first"})]
            elif n in {3, 6}:
                q = next(q for q in packet["question_state"] if q["question_id"] == "main")
                assert q["status"] == ("evidence_received" if n == 3 else "pending_review")
                doc = next(
                    d
                    for d in packet["documents"]
                    if d["version_id"] == ("first" if n == 3 else "second")
                )
                pid, text = next(iter(doc["excerpts"].items()))
                actions = [
                    (
                        "update_question_evidence",
                        {
                            "question_id": "main",
                            "seen_revision": q["evidence_revision"],
                            "observation": observation(
                                text, "2023-10-13" if n == 3 else "2023-10-14", passage_id=pid
                            ),
                        },
                    )
                ]
            elif n == 4:
                assert packet["question_state"][0]["status"] == "answered_provisionally"
                actions = [
                    (
                        "search",
                        {
                            "query": "Second Aurora Birch",
                            "purpose": "Check another record",
                            "intent": "source_trace",
                            "step_id": "main",
                            "limit": 1,
                        },
                    )
                ]
            elif n == 5:
                actions = [("read_document", {"version_id": "second"})]
            elif n == 7:
                assert packet["question_state"][0]["status"] == "conflicted"
                actions = [
                    (
                        "publish_plan",
                        {
                            "steps": [
                                {
                                    "step_id": "status",
                                    "title": "Model says done",
                                    "status": "completed",
                                }
                            ]
                        },
                    )
                ]
            else:
                assert n == 8 and packet["question_state"][0]["status"] == "conflicted"
                actions = []
                content = "错误初稿：批准证明收购已经完成。"
            for i, (name, args) in enumerate(actions):
                calls.append({"name": name, "args": args, "id": f"c2-{n}-{i}", "type": "tool_call"})
        message = AIMessage(
            content=content,
            tool_calls=calls,
            response_metadata={"finish_reason": "tool_calls" if calls else "stop"},
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_sdk_evidence_delivery_updates_questions_reopens_conflict_and_controls_final_answer(
    tmp_path, monkeypatch
):
    docs = [
        DocumentVersion(
            version_id=vid,
            title=f"{label} Aurora Birch",
            url=f"https://issuer.example/{vid}",
            content=f"On October {day}, 2023, Aurora completed its acquisition of Birch.",
        )
        for vid, label, day in [("first", "First", "13"), ("second", "Second", "14")]
    ]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([d.model_dump(mode="json") for d in docs]))
    monkeypatch.setattr(harness, "FixtureChatModel", ConflictScenario)
    run = tmp_path / "run"
    result = asyncio.run(
        harness.run_research(
            run,
            task_at().model_copy(update={"synthetic": True}),
            path,
            BudgetConfig(),
            mock=True,
            visual_progress=True,
        )
    )
    assert result["model_finalized"] and result["report_revision"]["status"] == "applied", result
    state = read_json(run / "question_state.json")
    main = state["questions"]["main"]
    assert main["status"] == "conflicted" and main["gap_status"] == "open_conflict"
    assert len(state["bindings"]) == 2  # final duplicate cannot close the existing conflict
    assert read_json(run / "answer_state.json")["inference"]["status"] == "unknown"
    assert "错误初稿" not in (run / "report.md").read_text()
    assert result["budget"]["model_calls"] == 9 and result["budget"]["tokens"] == 1080
    events = [json.loads(line) for line in (run / "trajectory.jsonl").read_text().splitlines()]
    transitions = [e for e in events if e["kind"] == "question_state_changed"]
    assert [e["after_status"] for e in transitions if e["question_id"] == "main"] == [
        "evidence_received",
        "answered_provisionally",
        "pending_review",
        "conflicted",
    ]
    actions = {e["action_id"] for e in events if e["kind"] == "tool_dispatch"}
    assert all(e["action_id"] in actions for e in transitions)
    snap = build_snapshot(run)
    assert snap["question_state"]["questions"]["main"]["status"] == "conflicted"
    assert any(n["provenance"] == "runtime_question_transition" for n in snap["nodes"])
