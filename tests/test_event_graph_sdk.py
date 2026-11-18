"""Scripted real-SDK acceptance for the opt-in P1 generic graph workflow."""

import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import news_agent.harness as harness
from news_agent.mock_model import FixtureChatModel
from news_agent.schemas import BudgetConfig, DocumentVersion, ResearchTask
from news_agent.storage import read_json, write_json


class GraphScenario(FixtureChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        actions, text = [], ""
        if messages[0].text.startswith("REPORT_REVISION:"):
            payload = json.loads(messages[-1].text)
            assert "output_schema" not in payload
            text = payload["draft"]
        elif self.calls == 1:
            assert "record_event_finding" in {t["function"]["name"] for t in kwargs["tools"]}
            assert "at most 350 words" not in messages[0].text
            actions = [
                (
                    "set_investigation_scope",
                    {
                        "scope": {
                            "direction": "backward",
                            "anchor": "User supplies the later correction",
                            "objective": "Trace the earlier statement and correction",
                        }
                    },
                ),
                (
                    "search",
                    {
                        "query": "Aurora statement correction",
                        "purpose": "Find the two original statements",
                        "intent": "initial",
                        "limit": 2,
                    },
                ),
            ]
        elif self.calls == 2:
            actions = [
                ("read_document", {"version_id": "early"}),
                ("read_document", {"version_id": "later"}),
            ]
        else:
            packet = next(
                json.loads(m.text.split("\n\n", 1)[1])
                for m in messages
                if m.type == "human" and m.text.startswith("Here is a summary")
            )
            assert packet["event_graph"]["scope"]["direction"] == "backward"
            originals = {
                d["version_id"]: next(iter(d["excerpts"].items())) for d in packet["documents"]
            }
            if self.calls == 3:
                for vid, when in [("early", "2026-11-01"), ("later", "2026-11-02")]:
                    pid, quote = originals[vid]
                    actions.append(
                        (
                            "record_event_finding",
                            {
                                "node_id": vid,
                                "title": vid,
                                "subject": "Aurora",
                                "kind": "statement",
                                "importance": "major",
                                "importance_reason": "Changes interpretation of the event",
                                "finding": {
                                    "assertion": quote,
                                    "stance": "supports",
                                    "occurrence": "statement",
                                    "event_time": {"precision": "day", "start": when},
                                    "originals": [{"passage_id": pid, "quote": quote}],
                                },
                            },
                        )
                    )
                actions.append(
                    (
                        "record_event_finding",
                        {
                            "node_id": "why",
                            "title": "Underlying cause",
                            "subject": "Aurora",
                            "kind": "hypothesis",
                            "importance": "minor",
                            "importance_reason": "Could explain the scope change",
                            "finding": {
                                "assertion": "Was a supplier change involved?",
                                "stance": "unknown",
                                "occurrence": "unknown",
                                "uncertainty": "No direct causal record collected",
                            },
                        },
                    )
                )
            elif self.calls == 4:
                pid, quote = originals["later"]
                actions = [
                    (
                        "record_event_relation",
                        {
                            "first": "later",
                            "second": "early",
                            "relation": "corrects",
                            "reason": "Original explicitly corrects the earlier statement",
                            "originals": [{"passage_id": pid, "quote": quote}],
                        },
                    )
                ]
            elif self.calls == 5:
                actions = [("inspect_event_graph", {"as_of": "2026-11-01"})]
            else:
                assert self.calls == 6
                text = (
                    f"The later statement corrects the earlier scope [{originals['later'][0]}]. "
                    f"The original statement stays in the history [{originals['early'][0]}]. "
                    "The underlying cause remains an important unknown."
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
                                "id": f"graph-{self.calls}-{i}",
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


def fixture(root):
    path = root / "corpus.json"
    write_json(
        path,
        [
            d.model_dump(mode="json")
            for d in [
                DocumentVersion(
                    version_id="early",
                    title="Aurora statement",
                    url="https://issuer.example/early",
                    content="On November 1, 2026, Aurora said all batches were included.",
                ),
                DocumentVersion(
                    version_id="later",
                    title="Aurora correction",
                    url="https://issuer.example/later",
                    content=(
                        "On November 2, 2026, Aurora corrected its statement: "
                        "only some batches were included."
                    ),
                ),
            ]
        ],
    )
    task = ResearchTask(
        task_id="graph-sdk",
        event="Aurora statements",
        question="Given the later correction, how did the event develop?",
        mode="retrospective",
        synthetic=True,
    )
    return task, path


def test_graph_workflow_uses_actual_sdk_tools_context_budget_and_keeps_temporal_history(
    tmp_path, monkeypatch
):
    task, path = fixture(tmp_path)
    monkeypatch.setattr(harness, "FixtureChatModel", GraphScenario)
    run = tmp_path / "run"
    result = asyncio.run(
        harness.run_research(run, task, path, BudgetConfig(), mock=True, workflow="event-map")
    )
    assert result["model_finalized"] and result["report_revision"]["status"] == "applied", result
    assert result["budget"]["model_calls"] == 7 and result["budget"]["tokens"] == 840
    graph = read_json(run / "event_graph.json")
    assert len(graph["nodes"]) == 3 and graph["relations"][0]["relation"] == "corrects"
    assert (
        not (run / "question_state.json").exists() and not (run / "repair_execution.json").exists()
    )
    cache = read_json(run / "tool_cache.json")
    prior = json.loads(cache["graph-5-0"]["content"])["event_graph"]
    assert (
        next(n for n in prior["nodes"] if n["node_id"] == "early")["status"] == "source_supported"
    )
    assert next(n for n in prior["nodes"] if n["node_id"] == "later")["status"] == "later_context"
    assert next(n for n in prior["nodes"] if n["node_id"] == "why")["status"] == "hypothesis"
    assert "important unknown" in (run / "report.md").read_text()
    before = (run / "trajectory.jsonl").read_text()
    resumed = asyncio.run(harness.run_research(run, resume=True))
    assert resumed == result and (run / "trajectory.jsonl").read_text() == before


def test_default_cli_mock_graph_mode_progresses_past_scope_into_originals(
    tmp_path, task, corpus_path
):
    result = asyncio.run(
        harness.run_research(
            tmp_path, task, corpus_path, BudgetConfig(), mock=True, workflow="event-map"
        )
    )
    graph = read_json(tmp_path / "event_graph.json")
    assert result["model_finalized"] and result["budget"]["search_rounds"] >= 1, result
    assert graph["nodes"] and len(graph["scope_history"]) == 1
    delivered = set(result["budget"]["delivered_passages"])
    assert all(
        ref["passage_id"] in delivered
        for node in graph["nodes"].values()
        for finding in node["findings"]
        for ref in finding["references"]
    )
