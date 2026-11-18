"""Actual SDK + synthetic retrieval: audit -> original context -> search -> bind -> re-audit."""

import asyncio
import copy
import json

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from news_agent.adaptive import PROMPT, AdaptiveInvestigation
from news_agent.budget import Budget, BudgetCallback, mock_usage
from news_agent.core_context import CoreEvidenceContext
from news_agent.event_graph import EventGraph, Finding, OriginalRef, Scope, graph_tools
from news_agent.event_report import Selection, choices, render, timeline_projection
from news_agent.harness import ResearchBoundary, make_tools, remember_passages
from news_agent.mock_model import FixtureChatModel
from news_agent.progress import document_locator, visual_search_tool
from news_agent.retrieval import LocalCorpus
from news_agent.schemas import BudgetConfig, DocumentVersion, ResearchTask
from news_agent.storage import EvidenceStore, RunStore, read_json
from news_agent.temporal_repair import complete_temporal_cycle
from news_agent.temporal_review import fixture_review, run_review


class RepairScenario(FixtureChatModel):
    step: int = 0
    outcome: str = "success"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text, actions = "", []
        if messages[0].text.startswith("TEMPORAL_REVIEW:"):
            payload = json.loads(messages[-1].text)
            value = fixture_review(payload)
            for r, d in zip(payload["findings"], value["decisions"], strict=True):
                if r["valid_time"]["precision"] == "unknown":
                    d["applicability"] = "needs_evidence"
                if self.outcome == "wrong_event":
                    for repair in d["repairs"]:
                        repair.update(
                            same_event_or_claim=False,
                            event_date_supported=False,
                            applicability="needs_evidence",
                            reason="Synthetic: another subject's rule; not the same event.",
                        )
            text = json.dumps(value)
        else:
            info = next(
                json.loads(m.text.split("\n", 1)[1])
                for m in reversed(messages)
                if m.text.startswith("ADAPTIVE_INVESTIGATION:\n")
            )
            targets = info.get("temporal_repair", {}).get("targets", [])
            names = {t["function"]["name"] for t in kwargs.get("tools", [])}
            if not names or not targets:
                text = (
                    "Synthetic result: preserve unresolved time or report the audited replacement."
                )
            else:
                t = targets[0]
                keys = [x["target_id"] for x in targets]
                gid = next(g["gap_id"] for g in info["queue"] if g["kind"] == "temporal_audit")
                if self.step == 0:
                    actions = [("begin_temporal_context", {"gap_id": gid})]
                elif self.step == 1:
                    actions = [("find_saved_text", {"version_id": "initial", "query": "rule"})]
                elif self.step == 2:
                    # Both local reads return already-delivered evidence. A direct
                    # supplementary search must still work without a close detour.
                    actions = [
                        (
                            "find_saved_text",
                            {
                                "version_id": "initial",
                                "query": "rule" if self.outcome == "repeat_context" else "June",
                            },
                        )
                    ]
                elif self.step == 3:
                    if self.outcome == "repeat_context":
                        assert "find_saved_text" not in names
                        assert (
                            info["temporal_repair"]["last_action_feedback"]["tool"]
                            == "find_saved_text"
                        )
                    actions = [
                        (
                            "search_gap",
                            {
                                "gap_id": gid,
                                "query": "dated original rule",
                                "strategy": "event_date",
                                "window": {"precision": "unknown"},
                                "movement": "unbounded",
                                "reason": "Find the missing event and applicability dates.",
                            },
                        )
                    ]
                elif self.step == 4:
                    actions = [("read_document", {"version_id": "dated"})]
                elif self.step == 5:
                    packet = next(
                        json.loads(m.text.split("\n\n", 1)[1])
                        for m in messages
                        if m.text.startswith("Here is a summary")
                    )
                    doc = next(d for d in packet["documents"] if d["version_id"] == "dated")
                    pid, quote = next(iter(doc["excerpts"].items()))
                    actions = [
                        (
                            "append_event_finding",
                            {
                                "node_id": t["node_id"],
                                "expected_revision": t["current_revision"],
                                "finding": {
                                    "assertion": "Aurora announced the rule on May 5, 2024.",
                                    "stance": "supports",
                                    "occurrence": "statement",
                                    "event_time": {"precision": "day", "start": "2024-05-05"},
                                    "valid_time": {
                                        "precision": "range",
                                        "start": "2024-06-01",
                                        "end": "2024-06-30",
                                    },
                                    "originals": [{"passage_id": pid}],
                                },
                            },
                        )
                    ]
                elif self.step == 6 and self.outcome == "yearless":
                    actions = [
                        (
                            "retain_temporal_unknown",
                            {
                                "target_ids": keys,
                                "reason": "Both originals omit the year; preserve unknown.",
                            },
                        )
                    ]
                elif self.step == 6:
                    # Durable candidates survive actual SDK context compaction.
                    fid = next(
                        r["finding_id"]
                        for r in info["temporal_repair"]["replacement_candidates"]
                        if r["event_time"]["precision"] == "day"
                    )
                    actions = [
                        (
                            "propose_temporal_resolution",
                            {
                                "target_ids": keys,
                                "replacement_finding_id": fid,
                                "reason": "New original states the rule and its period.",
                            },
                        )
                    ]
                elif self.step == 7 and self.outcome != "yearless":
                    actions = [
                        (
                            "request_temporal_reaudit",
                            {"reason": "New original and links are ready for review."},
                        )
                    ]
                else:
                    if info["active_round"]:
                        actions = [
                            (
                                "close_gap_round",
                                {"reason": "No further date basis; preserve uncertainty."},
                            )
                        ]
                    else:
                        actions = [
                            (
                                "retain_temporal_unknown",
                                {
                                    "target_ids": keys,
                                    "reason": "Another event; original time remains unresolved.",
                                },
                            )
                        ]
                self.step += 1
        msg = AIMessage(
            content=text,
            tool_calls=[
                {"name": n, "args": a, "id": f"repair-{self.step}-{i}"}
                for i, (n, a) in enumerate(actions)
            ],
            response_metadata={"finish_reason": "stop"},
        )
        msg.usage_metadata = mock_usage(messages, text + str(actions))
        return ChatResult(generations=[ChatGeneration(message=msg)])


def prepared(root, outcome="success"):
    task = ResearchTask(
        task_id="time-loop",
        event="Aurora rule",
        question="When was the rule announced and applicable?",
        mode="retrospective",
        synthetic=True,
    )
    e = EvidenceStore(RunStore(root))
    b = Budget(
        BudgetConfig(
            max_tokens=None,
            max_model_calls=None,
            max_tool_calls=None,
            deadline_seconds=None,
            max_output_tokens=16000,
        ),
        e.store,
    )
    g = EventGraph(task, e, b)
    g.set_scope(
        Scope(direction="both", anchor="Rule", objective=task.question, minimum_event_anchors=0), 0
    )
    doc = DocumentVersion(
        version_id="initial",
        title="Initial rule",
        url="https://issuer.example/initial",
        content="On May 5, Aurora announced the rule. The rule applies in June.",
    )
    p = e.add_document(doc)[0]
    remember_passages(b, [p.passage_id])
    g.record(
        "q",
        "Rule time",
        "Aurora",
        "question",
        "major",
        "Core temporal question",
        Finding(
            assertion="Aurora announced the rule, which applies in June.",
            stance="supports",
            occurrence="statement",
            originals=[OriginalRef(passage_id=p.passage_id, quote=p.text)],
            answers_node_question=True,
        ),
        0,
    )
    e.event_graph = g
    a = AdaptiveInvestigation(g, b)
    e.adaptive = a
    dated = "On May 5, 2024, Aurora announced the rule. It applies from 2024-06-01 to 2024-06-30."
    if outcome == "wrong_event":
        dated = dated.replace("Aurora", "Borealis")
    if outcome == "yearless":
        dated = "On May 5, Aurora announced the rule. It applies in June."
    corpus = LocalCorpus(
        [
            doc,
            DocumentVersion(
                version_id="dated",
                title="Dated original rule",
                url="https://issuer.example/dated",
                content=dated,
            ),
        ],
        task,
    )
    tools = [
        *make_tools(corpus, e, b, stop_guard=lambda: False),
        *graph_tools(g),
        document_locator(e, b),
    ]
    tools = [visual_search_tool(t, e, b) if t.name == "search" else t for t in tools]
    a.search = next(t for t in tools if t.name == "search")
    tools += a.tools()
    return task, e, b, g, a, tools


@pytest.mark.parametrize("outcome", ["success", "wrong_event", "yearless", "repeat_context"])
def test_actual_sdk_repair_cycle_and_date_guard(tmp_path, outcome):
    async def execute():
        task, e, b, g, a, tools = prepared(tmp_path, outcome)
        model = RepairScenario(outcome=outcome)
        context = CoreEvidenceContext(model, task, e, b)
        original = copy.deepcopy(g.data["nodes"]["q"]["findings"][0])
        boundary = ResearchBoundary(b, e.store, {t.name for t in tools}, evidence=e)
        graph = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=PROMPT,
            middleware=[context, boundary],
            checkpointer=InMemorySaver(),
        )
        cfg = {
            "configurable": {"thread_id": "test"},
            "callbacks": [BudgetCallback(b, 0)],
            "recursion_limit": 80,
        }
        b.finish_research("adaptive_gaps_resolved_or_stalled")
        await complete_temporal_cycle("Saved draft", task, context, model, b, cfg, graph)
        assert g.data["nodes"]["q"]["findings"][0] == original
        state = a.temporal_repair.data
        assert len(state["targets"]) == 2
        assert {t["status"] for t in state["targets"].values()} == (
            {"resolved"} if outcome in {"success", "repeat_context"} else {"retained_unknown"}
        )
        assert b.state["search_rounds"] == 1 and not b.state["unknown_usage_calls"]
        events = [
            json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
        ]
        assert any(e["kind"] == "temporal_context_started" for e in events)
        assert any(e["kind"] == "tool_dispatch" and e["name"] == "find_saved_text" for e in events)
        assert len(list((tmp_path / "temporal_reviews").glob("*/record.json"))) == (
            1 if outcome == "yearless" else 2
        )
        if outcome == "yearless":
            assert all(
                r["finding"]["event_time"]["precision"] == "unknown"
                for r in g.data["nodes"]["q"]["findings"]
            )
        return b.state["model_calls"]

    assert asyncio.run(execute()) > 2


def test_budget_resume_does_not_reset_usage_or_override_resource_stop(tmp_path):
    _, _, b, _, _, _ = prepared(tmp_path)
    b.state.update(
        tokens=1234,
        model_calls=5,
        tool_calls=3,
        phase="writing",
        stop_reason="research_budget_exhausted",
    )
    assert not b.reopen_temporal_research()
    b.state["stop_reason"] = "adaptive_gaps_resolved_or_stalled"
    assert b.reopen_temporal_research()
    assert (b.state["tokens"], b.state["model_calls"], b.state["tool_calls"]) == (1234, 5, 3)


def test_same_evidence_and_proposal_cannot_repeat_paid_audit(tmp_path):
    async def execute():
        task, e, b, g, a, _ = prepared(tmp_path)
        model = RepairScenario()
        context = CoreEvidenceContext(model, task, e, b)
        plan = choices(e, context.report_packet())
        cfg = {"callbacks": [BudgetCallback(b, 0)]}
        first = await run_review(task, e, plan, model, b, cfg)
        a.temporal_repair.ingest(first)
        count = b.state["model_calls"]
        again = await run_review(task, e, plan, model, b, cfg)
        assert first == again and count == b.state["model_calls"]
        assert read_json(tmp_path / "temporal_repair.json")["reviews"] == [first["binding"]]

    asyncio.run(execute())


def test_reused_context_keeps_distinct_lookups_and_search_available(tmp_path):
    async def execute():
        task, e, b, _, a, tools = prepared(tmp_path)
        review = await run_review(
            task,
            e,
            choices(e, CoreEvidenceContext(RepairScenario(), task, e, b).report_packet()),
            RepairScenario(),
            b,
            {"callbacks": [BudgetCallback(b, 0)]},
        )
        repair = a.temporal_repair
        repair.ingest(review)
        assert repair.start()
        gid = next(g["gap_id"] for g in a.data["gaps"].values() if g["kind"] == "temporal_audit")
        repair.begin_context(gid)
        r = a.active()
        r["local_no_gain"] = 2
        assert {"find_saved_text", "get_passage", "search_gap"} <= {
            t.name for t in a.surface(tools)
        }
        lookup = {
            "id": "context",
            "name": "find_saved_text",
            "args": {"version_id": "initial", "query": "June"},
        }
        assert a.before_tool(lookup) is None
        original = next(iter(e.passages.values()))
        a.after_tool(lookup, json.dumps({"passages": [original.model_dump(mode="json")]}))
        assert repair.summary()["recent_context"][0]["text"] == original.text
        assert a.before_tool({**lookup, "id": "repeat"}) is not None
        # Live v1.1 failure: compaction erased the denial, so the model repeated
        # identical calls. The durable prompt AND surface must now change.
        state = json.loads(a.prompt().split("\n", 1)[1])
        assert state["temporal_repair"]["last_action_feedback"]["arguments"] == lookup["args"]
        available = {t.name for t in a.surface(tools)}
        assert "find_saved_text" not in available
        assert {"get_passage", "search_gap"} <= available
        restored = type(repair)(a)
        assert restored.data["last_action_feedback"] == repair.data["last_action_feedback"]
        opened = {
            "id": "url",
            "name": "open_url",
            "args": {"url": "https://issuer.example/initial", "focus": "rule"},
        }
        assert a.before_tool(opened) is None
        a.after_tool(opened, json.dumps({"already_read": True}))
        assert a.before_tool({**opened, "id": "url-again"}) is not None
        assert "open_url" not in {t.name for t in a.surface(tools)}
        a.close_round("Saved context lacks dates")
        with pytest.raises(ValueError, match="already attempted"):
            repair.begin_context(gid)
        e.add_document(
            DocumentVersion(
                version_id="new",
                title="New context",
                url="https://issuer.example/new",
                content="Additional original context.",
            )
        )
        repair.begin_context(gid)
        assert a.active()["context_signature"] != r["context_signature"]
        assert {"find_saved_text", "get_passage"} <= {t.name for t in a.surface(tools)}
        assert not a.active().get("blocked_lookup_tools")

    asyncio.run(execute())


@pytest.mark.parametrize("mixed_targets", [False, True])
def test_existing_dated_candidate_clarifies_without_backfilling_and_can_be_revoked(
    tmp_path, mixed_targets
):
    async def execute():
        task, e, b, g, a, _ = prepared(tmp_path)
        original = copy.deepcopy(g.data["nodes"]["q"]["findings"][0])
        passage = e.add_document(
            DocumentVersion(
                version_id="already-saved",
                title="Separate dated record",
                url="https://issuer.example/dated-record",
                content=(
                    "On May 5, 2024, Aurora announced the rule. "
                    "It applies from 2024-06-01 to 2024-06-30."
                ),
            )
        )[0]
        remember_passages(b, [passage.passage_id])
        g.append(
            "q",
            Finding(
                assertion="Aurora announced the rule on May 5, 2024.",
                stance="supports",
                occurrence="statement",
                event_time={"precision": "day", "start": "2024-05-05"},
                valid_time={"precision": "range", "start": "2024-06-01", "end": "2024-06-30"},
                originals=[OriginalRef(passage_id=passage.passage_id, quote=passage.text)],
            ),
            1,
        )
        candidate = g.data["nodes"]["q"]["findings"][-1]["finding_id"]
        if mixed_targets:
            g.record(
                "other",
                "Another question",
                "Borealis",
                "question",
                "major",
                "Distinct question must not share a replacement",
                Finding.model_validate(original["finding"]),
                0,
            )
        model = RepairScenario()
        context = CoreEvidenceContext(model, task, e, b)
        cfg = {"callbacks": [BudgetCallback(b, 0)]}
        review = await run_review(task, e, choices(e, context.report_packet()), model, b, cfg)
        repair = a.temporal_repair
        repair.ingest(review)
        assert repair.start()
        targets = repair.summary()["targets"]
        assert all(
            (candidate in t["eligible_saved_candidate_ids"]) == (t["node_id"] == "q")
            for t in targets
        )
        receipt = repair.propose(
            [t["target_id"] for t in targets], candidate, "Same rule; separate dated original."
        )
        assert len(receipt["accepted_target_ids"]) == 2
        assert len(receipt["rejected_targets"]) == (2 if mixed_targets else 0)
        a.after_tool(
            {"name": "propose_temporal_resolution", "args": {}, "id": "proposal"},
            json.dumps(receipt),
        )
        assert (
            repair.summary()["last_action_feedback"]["accepted_target_ids"]
            == receipt["accepted_target_ids"]
        )
        assert all(
            t["status"] == ("awaiting_review" if t["node_id"] == "q" else "open")
            for t in repair.data["targets"].values()
        )
        reviewed = await run_review(task, e, choices(e, context.report_packet()), model, b, cfg)
        repair.ingest(reviewed)
        assert all(
            t["status"] == ("resolved" if t["node_id"] == "q" else "open")
            for t in repair.data["targets"].values()
        )
        assert g.data["nodes"]["q"]["findings"][0] == original
        assert original["finding"]["event_time"]["precision"] == "unknown"
        plan = choices(e, context.report_packet())
        plan["temporal_review"] = reviewed
        selected = Selection(
            summary=[r["finding_id"] for r in plan["graph_choices"]],
            timeline=[r["finding_id"] for r in plan["graph_choices"]],
        )
        projection = timeline_projection(selected, plan)
        assert projection["clarified"] == [original["finding_id"]]
        assert original["finding_id"] not in projection["undated"]
        assert "已由补充证据澄清的原记录" in render(selected, task, e, plan)
        assert f"{original['finding_id']}：日期仍需补证" not in render(selected, task, e, plan)
        stale = copy.deepcopy(plan)
        for target in stale["temporal_repairs"]:
            target["review_binding"] = "stale-audit"
        assert not timeline_projection(selected, stale)["clarified"]
        # A later different audit may withdraw a prior decision. Never leave a
        # stale positive link in the report/UI after reopening that target.
        withdrawn = copy.deepcopy(reviewed)
        withdrawn["binding"] = "synthetic-new-audit-input"
        for d in withdrawn["decisions"]:
            if d["finding_id"] == candidate:
                d.update(event_verdict="conflicting", applicability="needs_evidence")
                for relation in d["repairs"]:
                    relation.update(event_date_supported=False, applicability="needs_evidence")
        repair.ingest(withdrawn)
        assert all(
            t["status"] == "open" and not t.get("resolved_by")
            for t in repair.data["targets"].values()
        )
        assert b.state["search_rounds"] == 0

    asyncio.run(execute())
