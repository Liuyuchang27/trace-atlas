"""Engineering-only P2 CLI fixture; never answers a real user's question."""

import json

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .budget import mock_usage
from .mock_model import FixtureChatModel


class AdaptiveFixtureModel(FixtureChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if messages[0].text.startswith("TEMPORAL_REVIEW:"):
            return super()._generate(messages, stop, run_manager, **kwargs)
        self.calls += 1
        tools = {t["function"]["name"] for t in kwargs.get("tools", [])}
        actions, text = [], ""
        if messages[0].text.startswith("REPORT_REVISION:"):
            payload = json.loads(messages[-1].text)
            if "graph_choices" in payload:
                from .event_report import fixture_selection

                text = json.dumps(fixture_selection(payload))
            else:
                text = payload["draft"]
        else:
            info = next(
                json.loads(m.text.split("\n", 1)[1])
                for m in reversed(messages)
                if m.text.startswith("ADAPTIVE_INVESTIGATION:\n")
            )
            packets = [
                json.loads(m.text.split("\n\n", 1)[1])
                for m in messages
                if m.text.startswith("Here is a summary")
            ]
            packet = packets[-1] if packets else {}
            graph = packet.get("event_graph", {})
            if not graph:
                for m in reversed(messages):
                    if m.type == "tool":
                        try:
                            value = json.loads(m.text).get("event_graph", {})
                        except (ValueError, AttributeError):
                            continue
                        if "scope" in value:
                            graph = value
                            break
            if "begin_investigation" in tools:
                actions = [
                    (
                        "begin_investigation",
                        {
                            "scope": {
                                "direction": "both",
                                "anchor": "Synthetic fixture",
                                "objective": "Exercise search/read/state tools only",
                            },
                            "node_id": "fixture-question",
                            "title": "Synthetic fixture question",
                        },
                    )
                ]
            elif "search_gap" in tools and info["queue"]:
                gap = info["queue"][0]
                query = "Aurora Birch original records " + gap["gap_id"] + str(gap["attempts"])
                actions = [
                    (
                        "search_gap",
                        {
                            "gap_id": gap["gap_id"],
                            "query": query,
                            "strategy": "clarification",
                            "window": {"precision": "unknown"},
                            "movement": "unbounded",
                            "reason": "SCRIPTED: exercise linked supplementary retrieval",
                        },
                    )
                ]
            elif info["active_round"] and "read_document" in tools:
                r = info["active_round"]
                if not r["actions"] and r["search_result"]["candidate_ids"]:
                    actions = [
                        ("read_document", {"version_id": r["search_result"]["candidate_ids"][0]})
                    ]
                else:
                    node = next(
                        (n for n in graph.get("nodes", []) if n["node_id"] == r["node_id"]), None
                    )
                    docs = [d for d in packet.get("documents", []) if d["excerpts"]]
                    if node and docs and "append_event_finding" in tools:
                        pid, quote = next(iter(docs[0]["excerpts"].items()))
                        ref = {"passage_id": pid}
                        if len(quote) > 900:
                            ref["excerpt_id"] = docs[0]["binding_choices"][pid][0]["excerpt_id"]
                        actions.append(
                            (
                                "append_event_finding",
                                {
                                    "node_id": "fixture-question",
                                    "expected_revision": node["revision"],
                                    "finding": {
                                        "assertion": "SCRIPTED context, no answer assessment",
                                        "stance": "context",
                                        "occurrence": "unknown",
                                        "originals": [ref],
                                    },
                                },
                            )
                        )
                    actions.append(
                        (
                            "close_gap_round",
                            {"reason": "SCRIPTED context only, no factual conclusion"},
                        )
                    )
            else:
                text = (
                    "SCRIPTED SYNTHETIC FIXTURE. Retrieval and state tools exercised. "
                    "No real question answered; factual and temporal gaps remain unresolved."
                )
        message = AIMessage(
            content=text,
            tool_calls=[
                {
                    "name": n,
                    "args": a,
                    "id": f"adaptive-fixture-{self.calls}-{i}",
                    "type": "tool_call",
                }
                for i, (n, a) in enumerate(actions)
            ],
            response_metadata={"finish_reason": "tool_calls" if actions else "stop"},
        )
        message.usage_metadata = mock_usage(messages, message.text)
        return ChatResult(generations=[ChatGeneration(message=message)])
