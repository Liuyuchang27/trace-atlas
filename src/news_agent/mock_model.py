"""A scripted model for SDK integration tests, never a research-quality baseline."""

from __future__ import annotations

import json
import re
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from .budget import mock_usage


class FixtureChatModel(BaseChatModel):
    profile: dict = {"max_input_tokens": 100000, "max_output_tokens": 1500}
    calls: int = 0
    auxiliary_calls: int = 0
    graph_seen: list[str] = []

    @property
    def _llm_type(self):
        return "fixture-script-not-for-evaluation"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(
            tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice, **kwargs
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if "tools" not in kwargs:
            self.auxiliary_calls += 1
        if messages[0].text.startswith("TEMPORAL_REVIEW:"):
            from .temporal_review import fixture_review

            message = AIMessage(
                content=json.dumps(fixture_review(json.loads(messages[-1].content))),
                response_metadata={"finish_reason": "stop"},
            )
            message.usage_metadata = mock_usage(messages, message.content)
            return ChatResult(generations=[ChatGeneration(message=message)])
        if messages[0].text.startswith("REPORT_REVISION:"):
            payload = json.loads(messages[-1].content)
            if "graph_choices" in payload:
                from .event_report import fixture_selection

                content = json.dumps(fixture_selection(payload))
            elif "output_schema" in payload:
                # Scripted fixture plumbing: no semantic or factual judgment.
                rows = [
                    {
                        "kind": "announcement",
                        "subject": text[:1],
                        "target_event": False,
                        "occurrence": "unknown",
                        "event_date": None,
                        "passage_id": pid,
                        "quote": text[:400],
                    }
                    for doc in payload["eligible_evidence"]["documents"]
                    for pid, text in list(doc["excerpts"].items())[:1]
                    if len(text) >= 8
                ][:3]
                content = json.dumps({"observations": rows, "legal_relations": [], "gaps": []})
            else:
                content = payload["draft"]
            message = AIMessage(content=content, response_metadata={"finish_reason": "stop"})
            message.usage_metadata = mock_usage(messages, message.content)
            return ChatResult(generations=[ChatGeneration(message=message)])
        if messages[0].text.startswith("MECHANISM_REVIEW:"):
            kind = messages[0].text.split("\n", 1)[0].split(":", 1)[1]
            payload = json.loads(messages[-1].content)
            reason = "SCRIPTED FIXTURE assessment; not semantic evaluation."
            if kind == "T":
                p = payload["passages"][0]
                text = p["excerpts"][0]["text"]
                found = re.search(r"\d{4}-\d{2}-\d{2}", text)
                value = {
                    "stage": "approval" if "approv" in text else "announcement",
                    "kind": "observed",
                    "event_date": found.group() if found else None,
                    "verdict": "supported",
                    "passage_id": p["passage_id"],
                    "excerpt_id": p["excerpts"][0]["excerpt_id"],
                    "reason": reason,
                }
            elif kind == "S":
                value = {
                    "links": [
                        {
                            "passage_id": p["passage_id"],
                            "relation": "supports"
                            if payload["claim"]["text"] in p["excerpts"][0]["text"]
                            else "insufficient",
                            "excerpt_id": p["excerpts"][0]["excerpt_id"],
                            "reason": reason,
                        }
                        for p in payload["passages"]
                    ]
                }
            else:
                docs = payload["documents"]
                explicit = any(
                    a["url"] in b["excerpts"][0]["text"] for a in docs for b in docs if a != b
                )
                value = {
                    "relation": "cites" if explicit else "unknown",
                    "confidence": "confirmed" if explicit else "unknown",
                    "evidence": [
                        {
                            "version_id": d["version_id"],
                            "excerpt_id": d["excerpts"][0]["excerpt_id"],
                        }
                        for d in docs
                    ],
                    "needs_independent_record": False,
                    "reason": reason,
                }
            message = AIMessage(
                content=json.dumps(value), response_metadata={"finish_reason": "stop"}
            )
            message.usage_metadata = mock_usage(messages, message.content)
            return ChatResult(generations=[ChatGeneration(message=message)])
        names = {t["function"]["name"] for t in kwargs.get("tools", [])}
        observations = []
        graph_packet = None
        for message in messages:
            if isinstance(message, ToolMessage):
                try:
                    observations.append(json.loads(message.content))
                except (ValueError, TypeError):
                    continue
        # Only the new core packet has packet_id. Legacy fixture behavior stays unchanged.
        for message in messages:
            if (
                message.type != "human"
                or not message.text.startswith("Here is a summary")
                or '"packet_id"' not in message.text
            ):
                continue
            packet = json.loads(message.text.split("\n\n", 1)[1])
            if "packet_id" not in packet:
                continue
            graph_packet = packet.get("event_graph")
            observations.append({"results": packet["unread_candidates"]})
            for doc in packet["documents"]:
                observations.append(
                    {
                        "version_id": doc["version_id"],
                        "passages": [
                            {"passage_id": pid, "text": text}
                            for pid, text in doc["excerpts"].items()
                        ],
                    }
                )
            observations.extend({"claim": c} for c in packet["claims"])
        graph_packet = graph_packet or next(
            (o["event_graph"] for o in reversed(observations) if "event_graph" in o), None
        )
        hits = [hit for o in observations for hit in o.get("results", [])]
        reads = [o for o in observations if "passages" in o]
        claims = [o["claim"] for o in observations if "claim" in o]
        known = {p for c in claims for p in c["passage_ids"]}
        unread = [h for h in hits if h["version_id"] not in {r["version_id"] for r in reads}]
        unrecorded = [p for r in reads for p in r["passages"] if p["passage_id"] not in known]
        if "record_event_finding" in names:
            action = None
            if not graph_packet or graph_packet.get("scope") is None:
                action = (
                    "set_investigation_scope",
                    {
                        "scope": {
                            "direction": "both",
                            "anchor": "Synthetic fixture input",
                            "objective": "Record supplied synthetic originals",
                        }
                    },
                )
            elif not hits and "search" in names:
                action = (
                    "search",
                    {
                        "query": "Aurora Birch",
                        "purpose": "Find fixture originals",
                        "intent": "initial",
                    },
                )
            elif unread and "read_document" in names:
                action = ("read_document", {"version_id": unread[0]["version_id"]})
            else:
                todo = [
                    p
                    for r in reads
                    for p in r["passages"]
                    if p["passage_id"] not in self.graph_seen
                ]
                if todo:
                    p = todo[0]
                    self.graph_seen.append(p["passage_id"])
                    action = (
                        "record_event_finding",
                        {
                            "node_id": p["passage_id"],
                            "title": "Synthetic original " + str(len(self.graph_seen)),
                            "subject": "Fixture event",
                            "kind": "statement",
                            "importance": "minor",
                            "importance_reason": "Fixture evidence record",
                            "finding": {
                                "assertion": "SCRIPTED original context, not a semantic assessment",
                                "stance": "context",
                                "occurrence": "statement",
                                "originals": [
                                    {"passage_id": p["passage_id"], "quote": p["text"][:400]}
                                ],
                            },
                        },
                    )
            msg = AIMessage(
                content=""
                if action
                else "SCRIPTED generic graph fixture; not a real event answer.",
                tool_calls=[
                    {
                        "name": action[0],
                        "args": action[1],
                        "id": "fixture-graph-" + uuid4().hex,
                        "type": "tool_call",
                    }
                ]
                if action
                else [],
                response_metadata={"finish_reason": "tool_calls" if action else "stop"},
            )
            msg.usage_metadata = mock_usage(messages, msg.content)
            return ChatResult(generations=[ChatGeneration(message=msg)])
        call = None
        if names:
            if not any("results" in o for o in observations) and "search" in names:
                task = next(
                    (
                        json.loads(m.content)["task"]
                        for m in messages
                        if m.type == "human" and m.content.startswith('{"task":')
                    ),
                    {"event": "Aurora Birch acquisition"},
                )
                call = ("search", {"query": task["event"], "limit": 5})
            elif unrecorded and "record_claim" in names:
                p = unrecorded[0]
                date = re.search(r"\d{4}-\d{2}-\d{2}", p["text"])
                call = (
                    "record_claim",
                    {
                        "text": p["text"],
                        "passage_ids": [p["passage_id"]],
                        "event_time": date.group() if date else None,
                    },
                )
            elif unread and "read_document" in names:
                call = ("read_document", {"version_id": unread[0]["version_id"]})
            else:
                state = next(
                    (
                        m.content
                        for m in reversed(messages)
                        if m.type == "human" and m.content.startswith("Investigation policy state")
                    ),
                    None,
                )
                if state:
                    policy = json.loads(state[state.index('{"condition":') :])
                    for recommendation in policy.get("recommended_actions", []):
                        if recommendation["tool"] in names:
                            name = recommendation["tool"]
                            arguments = {
                                k: v
                                for k, v in recommendation.items()
                                if k not in {"tool", "target"}
                            }
                            if name in {"focus_date_or_subevent", "seek_corroboration"}:
                                arguments["query"] = (
                                    "Aurora Birch acquisition closing completion evidence"
                                )
                            call = (name, arguments)
                            break
        forced = kwargs.get("tool_choice")
        if isinstance(forced, dict) and forced.get("function", {}).get("name"):
            name = forced["function"]["name"]
            if call is None or call[0] != name:
                state = next(
                    (
                        m.content
                        for m in reversed(messages)
                        if m.type == "human" and m.content.startswith("Investigation policy state")
                    ),
                    None,
                )
                if state:
                    policy = json.loads(state[state.index('{"condition":') :])
                    action = next(
                        (a for a in policy.get("recommended_actions", []) if a["tool"] == name),
                        None,
                    )
                    if action and name != "record_claim":
                        call = (
                            name,
                            {k: v for k, v in action.items() if k not in {"tool", "target"}},
                        )
        if call:
            name, arguments = call
            schema = next(
                (
                    t["function"].get("parameters", {})
                    for t in kwargs.get("tools", [])
                    if t["function"]["name"] == name
                ),
                {},
            )
            if name == "search" and "purpose" in schema.get("properties", {}):
                arguments.update(
                    purpose="SCRIPTED synthetic retrieval acceptance.", intent="initial"
                )
            message = AIMessage(
                content="", tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex}]
            )
        else:
            excerpts = "\n".join(
                f"- {c['text']} " + " ".join(f"[{p}]" for p in c["passage_ids"]) for c in claims
            )
            message = AIMessage(
                content=(
                    "# Offline integration demonstration\n\n"
                    "SYNTHETIC DATA + SCRIPTED MOCK MODEL. Not a factual evaluation result.\n\n"
                    "## Retrieved evidence excerpts\n\n"
                    + (excerpts or "No excerpts recorded.")
                    + "\n\n## Unresolved questions\n\n"
                    "A live model is required for answer synthesis and semantic assessment."
                )
            )
        message.usage_metadata = mock_usage(messages, message.content + str(message.tool_calls))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
