"""Real SDK and checkpoint path; synthetic actions are not factual quality labels."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_event_graph_sdk import fixture

import news_agent.harness as harness
from news_agent.mock_model import FixtureChatModel
from news_agent.schemas import BudgetConfig
from news_agent.storage import read_json


class AdaptiveScenario(FixtureChatModel):
    versions: dict = {}
    source_failure: bool = False
    automatic_round_close: bool = False
    yearless: bool = False
    date_recovery: bool = False

    def original(self, messages, vid):
        packet = next(
            json.loads(m.text.split("\n\n", 1)[1])
            for m in messages
            if m.type == "human" and m.text.startswith("Here is a summary")
        )
        doc = next(d for d in packet["documents"] if d["version_id"] == self.versions.get(vid, vid))
        pid, quote = next((pid, text) for pid, text in doc["excerpts"].items() if "Aurora" in text)
        when = "2026-11-01" if vid == "early" else "2026-11-02"
        return {
            "assertion": quote,
            "stance": "supports",
            "occurrence": "statement",
            "event_time": {"precision": "unknown"}
            if self.yearless
            else {"precision": "day", "start": when, "end": when},
            "originals": [{"passage_id": pid, "quote": quote}],
        }

    def create(self, vid, finding):
        return (
            "create_event_node",
            {
                "node_id": vid,
                "title": vid,
                "subject": "Aurora",
                "kind": "statement",
                "importance": "major",
                "importance_reason": "Changes scope chronology",
                "finding": {
                    **finding,
                    "originals": [{"passage_id": r["passage_id"]} for r in finding["originals"]],
                },
            },
        )

    def review(self, vid, finding):
        return (
            "review_node_sources",
            {
                "node_id": vid,
                "originals": finding["originals"],
                "reason": "Issuer's own statement; no independent certification",
            },
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        actions, text = [], ""
        # Revision instructions can follow a shared answer-first preamble.
        # Route by the dedicated system marker, never by research-turn count.
        markers = set(messages[0].text.splitlines())
        if markers & {"REPORT_REVISION:", "TEMPORAL_REVIEW:"}:
            payload = json.loads(messages[-1].text)
            if "TEMPORAL_REVIEW:" in markers:
                from news_agent.temporal_review import fixture_review

                text = json.dumps(fixture_review(payload))
            elif "graph_choices" in payload:
                from news_agent.event_report import fixture_selection

                text = json.dumps(fixture_selection(payload))
            else:
                text = payload["draft"]
        elif self.calls == 1:
            actions = [
                (
                    "begin_investigation",
                    {
                        "scope": {
                            "direction": "both",
                            "anchor": "Later correction",
                            "objective": "Check premise and changing statements",
                        },
                        "node_id": "premise",
                        "title": "Did the scope change?",
                    },
                )
            ]
        else:
            adaptive = next(
                json.loads(m.text.split("\n", 1)[1])
                for m in reversed(messages)
                if m.type == "human" and m.text.startswith("ADAPTIVE_INVESTIGATION:\n")
            )
            if self.date_recovery and self.calls == 8:
                recovery = next(
                    json.loads(m.text.split("\n", 1)[1])
                    for m in messages
                    if m.type == "human" and m.text.startswith("EVENT_WRITE_RECOVERY:\n")
                )
                actions = [
                    (
                        "resolve_event_write",
                        {
                            "recovery_id": recovery["recovery_id"],
                            "decision": "bind_unknown",
                            "expected_revision": recovery["current_revision"],
                            "reason": "Original gives month/day only; event year remains unknown.",
                            "assertion": (
                                "On November 2, Aurora corrected its statement: "
                                "only some batches were included."
                            ),
                            "date_text": "November 2",
                        },
                    )
                ]
            elif self.calls in {2, 5}:
                gid = next(
                    g["gap_id"]
                    for g in adaptive["queue"]
                    if g["node_id"] == "premise"
                    and g["kind"] == "question_or_explanation_unresolved"
                )
                when = "2026-11-01" if self.calls == 2 else "2026-11-02"
                actions += [
                    (
                        "search_gap",
                        {
                            "gap_id": gid,
                            "query": "Aurora statement"
                            if self.calls == 2
                            else "Aurora later correction",
                            "strategy": "primary_record" if self.calls == 2 else "counterevidence",
                            "window": {"precision": "day", "start": when, "end": when},
                            "movement": "initial" if self.calls == 2 else "shift_forward",
                            "reason": "Find originals for the exact scope-change question",
                        },
                    )
                ]
            elif self.calls in {3, 6}:
                vid = self.versions.get(
                    "early" if self.calls == 3 else "later", "early" if self.calls == 3 else "later"
                )
                assert vid in adaptive["active_round"]["search_result"]["candidate_ids"]
                actions = [("read_document", {"version_id": vid})]
            elif self.calls in {4, 7}:
                if self.calls != 7 or not self.source_failure:
                    f = self.original(messages, "early" if self.calls == 4 else "later")
                    if self.calls == 4:
                        f["stance"] = "context"
                    else:
                        f["answers_node_question"] = True
                    actions = [
                        (
                            "append_event_finding",
                            {
                                "node_id": "premise",
                                "expected_revision": 1 if self.calls == 4 else 2,
                                "finding": {
                                    **f,
                                    "originals": [
                                        {"passage_id": r["passage_id"]} for r in f["originals"]
                                    ],
                                },
                            },
                        )
                    ]
                if not self.automatic_round_close or self.source_failure:
                    actions.append(
                        (
                            "close_gap_round",
                            {
                                "reason": "Only first context"
                                if self.calls == 4
                                else "Later original bound or unavailable; preserve remaining gaps"
                            },
                        )
                    )
            elif self.calls == 8 and not self.source_failure and not self.yearless:
                later = self.original(messages, "later")
                early = self.original(messages, "early")
                actions = [
                    self.create("early", early),
                    self.review("early", early),
                    self.create("later", later),
                    (
                        "record_event_relation",
                        {
                            "first": "later",
                            "second": "early",
                            "relation": "corrects",
                            "originals": later["originals"],
                            "reason": "Explicit original correction",
                        },
                    ),
                    self.review("later", later),
                    self.review("premise", later),
                ]
                if candidates := adaptive.get("source_review_candidates"):
                    # The uncapped source phase uses saved bindings, not copied quotes.
                    actions = [
                        (
                            "review_bound_sources",
                            {
                                "node_id": "premise",
                                "finding_ids": sorted({r["finding_id"] for r in candidates}),
                                "reason": "Issuer's own statements; no independent certification",
                            },
                        )
                    ]
            else:
                assert self.calls == (
                    8 if self.source_failure or (self.yearless and not self.date_recovery) else 9
                )
                if not self.source_failure and not self.yearless:
                    assert not adaptive["queue"] and adaptive["completed_rounds"] == 2
                text = (
                    "SYNTHETIC: original unavailable, premise unresolved."
                    if self.source_failure
                    else (
                        "SYNTHETIC: later source corrects earlier scope; "
                        "no independent factual certification."
                    )
                )
                packet = next(
                    json.loads(m.text.split("\n\n", 1)[1])
                    for m in messages
                    if m.type == "human" and m.text.startswith("Here is a summary")
                )
                text += " " + " ".join(
                    f"[{pid}]" for d in packet["documents"] for pid in list(d["excerpts"])[:1]
                )
        if self.yearless:
            for i, (name, args) in enumerate(actions):
                if name == "append_event_finding":
                    f = args["finding"]
                    if self.date_recovery and self.calls == 7:
                        f["event_time"] = {"precision": "day", "start": "2026-11-02"}
                        continue  # deliberately refused; next turn must explicitly resolve it
                    f.pop("event_time", None)
                    f["date_text"] = "November 1" if self.calls == 4 else "November 2"
                    f["uncertainty"] = "The source omits the year; no inferred cutoff year."
                    actions[i] = ("append_undated_finding", args)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=text,
                        tool_calls=[
                            {
                                "name": n,
                                "args": a,
                                "id": f"p2-{self.calls}-{i}",
                                "type": "tool_call",
                            }
                            for i, (n, a) in enumerate(actions)
                        ],
                        response_metadata={"finish_reason": "tool_calls" if actions else "stop"},
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        },
                    )
                )
            ]
        )


class WritingRecoveryScenario(AdaptiveScenario):
    """One stale binding at writing boundary, then a fresh binding, within total budget."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.calls < 6 or messages[0].text.startswith(("REPORT_REVISION:", "TEMPORAL_REVIEW:")):
            return super()._generate(messages, stop, run_manager, **kwargs)
        self.calls += 1
        actions, text = [], ""
        if self.calls in {7, 8}:
            f = self.original(messages, "early" if self.calls == 7 else "later")
            f["answers_node_question"] = self.calls == 8
            actions = [
                (
                    "append_event_finding",
                    {
                        "node_id": "premise",
                        "expected_revision": next(
                            json.loads(m.text.split("\n", 1)[1])["target_revision"]
                            for m in reversed(messages)
                            if m.text.startswith("ADAPTIVE_INVESTIGATION:\n")
                        ),
                        "finding": {
                            **f,
                            "originals": [{"passage_id": r["passage_id"]} for r in f["originals"]],
                        },
                    },
                )
            ]
        else:
            assert self.calls == 9
            packet = next(
                json.loads(m.text.split("\n\n", 1)[1])
                for m in messages
                if m.type == "human" and m.text.startswith("Here is a summary")
            )
            text = "SYNTHETIC writing-boundary recovery, not factual certification. " + " ".join(
                f"[{pid}]" for d in packet["documents"] for pid in list(d["excerpts"])[:1]
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=text,
                        tool_calls=[
                            {
                                "name": n,
                                "args": a,
                                "id": f"writing-recovery-{self.calls}-{i}",
                                "type": "tool_call",
                            }
                            for i, (n, a) in enumerate(actions)
                        ],
                        response_metadata={"finish_reason": "tool_calls" if actions else "stop"},
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        },
                    )
                )
            ]
        )


class ConsecutiveWritingScenario(AdaptiveScenario):
    """Two same-node writes collide, then one explicit model correction or skip."""

    recovery_outcome: str = "success"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.calls < 6 or messages[0].text.startswith(("REPORT_REVISION:", "TEMPORAL_REVIEW:")):
            return super()._generate(messages, stop, run_manager, **kwargs)
        self.calls += 1
        actions, text = [], ""
        if self.calls == 7:
            later = self.original(messages, "later")
            context = {
                **later,
                "assertion": "Later record exists; answer not yet bound.",
                "stance": "context",
            }
            for finding in [context, {**later, "answers_node_question": True}]:
                actions.append(
                    (
                        "append_event_finding",
                        {
                            "node_id": "premise",
                            "expected_revision": 2,
                            "finding": {
                                **finding,
                                "originals": [
                                    {"passage_id": r["passage_id"]} for r in finding["originals"]
                                ],
                            },
                        },
                    )
                )
        elif self.calls == 8:
            recovery = next(
                json.loads(m.text.split("\n", 1)[1])
                for m in messages
                if m.type == "human" and m.text.startswith("EVENT_WRITE_RECOVERY:\n")
            )
            assert recovery["current_revision"] == 3
            actions = [
                (
                    "resolve_event_write",
                    {
                        "recovery_id": recovery["recovery_id"],
                        "decision": "skip" if self.recovery_outcome == "skip" else "retry",
                        "expected_revision": 2
                        if self.recovery_outcome == "failure"
                        else recovery["current_revision"],
                        "reason": "SCRIPTED explicit review of the saved rejected proposal",
                    },
                )
            ]
        else:
            assert self.calls == 9
            later = self.original(messages, "later")
            text = (
                "SYNTHETIC: later correction explicitly bound. "
                if self.recovery_outcome == "success"
                else "SYNTHETIC: correction unresolved after one recovery opportunity. "
            ) + " ".join(f"[{r['passage_id']}]" for r in later["originals"])
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=text,
                        tool_calls=[
                            {
                                "name": n,
                                "args": a,
                                "id": f"consecutive-{self.calls}-{i}",
                                "type": "tool_call",
                            }
                            for i, (n, a) in enumerate(actions)
                        ],
                        response_metadata={"finish_reason": "tool_calls" if actions else "stop"},
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        },
                    )
                )
            ]
        )


@pytest.mark.parametrize("automatic", [False, True])
def test_actual_sdk_two_rounds_same_gap_new_original_updates_node_and_window(
    tmp_path, monkeypatch, automatic
):
    task, path = fixture(tmp_path)
    monkeypatch.setattr(
        harness, "AdaptiveFixtureModel", lambda: AdaptiveScenario(automatic_round_close=automatic)
    )
    run = tmp_path / "run"
    result = asyncio.run(
        harness.run_research(
            run, task, path, BudgetConfig(), mock=True, workflow="adaptive-event-map"
        )
    )
    assert result["model_finalized"], result
    state = read_json(run / "adaptive_investigation.json")
    assert len(state["rounds"]) == 2
    assert state["rounds"][0]["gap_id"] == state["rounds"][1]["gap_id"]
    assert state["rounds"][1]["movement"] == "shift_forward"
    assert all(r["evidence_gain"] and r["status"] == "closed" for r in state["rounds"])
    graph = read_json(run / "event_graph.json")
    assert len(graph["nodes"]["premise"]["findings"]) == 3
    assert graph["relations"][0]["relation"] == "corrects"
    assert result["budget"]["search_rounds"] == 2
    assert result["budget"]["model_calls"] == 11
    events = [json.loads(line) for line in (run / "trajectory.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "adaptive_search_dispatch"]) == 2
    assert not [e for e in events if e["kind"] == "adaptive_tool_denied"]


def test_default_cli_fixture_uses_linked_gap_reads_and_does_not_claim_real_answers(tmp_path):
    from pathlib import Path

    from news_agent.schemas import ResearchTask

    root = Path(__file__).resolve().parents[1]
    task = ResearchTask.model_validate(read_json(root / "data/fixtures/task.json"))
    result = asyncio.run(
        harness.run_research(
            tmp_path / "run",
            task,
            root / "data/fixtures/corpus.json",
            BudgetConfig(),
            mock=True,
            workflow="adaptive-event-map",
        )
    )
    assert result["model_finalized"], result
    e = read_json(tmp_path / "run/evidence.json")
    a = read_json(tmp_path / "run/adaptive_investigation.json")
    assert e["documents"] and a["rounds"]
    assert result["status"] == "partial"
    assert "SCRIPTED" in (tmp_path / "run/report.md").read_text()
