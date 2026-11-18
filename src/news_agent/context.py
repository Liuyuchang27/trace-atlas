"""Bounded evidence context through the SDK summarization hook.

Selection uses document order and recent access, without event-stage or
source-independence judgments. Originals remain in the evidence store.
"""

from __future__ import annotations

import json

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

from .budget import Budget, BudgetExceeded, input_upper_estimate
from .schemas import ResearchTask
from .storage import EvidenceStore


def encoded(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def byte_size(value) -> int:
    return len(encoded(value).encode())


def context_policy(config) -> dict:
    limit = max(512, min(12000, config.reserve_tokens * 3 // 5))
    return {
        "method": "sdk_deterministic_evidence_packet",
        "packet_bytes": limit,
        "trigger_byte_upper_estimate": limit * 2,
        "writing_tool_turns": 2,
        "retention": "full_document_leads_then_recent_passages_and_claims_with_action_history",
        "writing_trigger": "compact_each_writing_turn_after_reserve_precheck",
    }


class EvidenceContext(SummarizationMiddleware):
    @property
    def name(self):
        return "SummarizationMiddleware"

    def __init__(self, model, task: ResearchTask, evidence: EvidenceStore, budget: Budget):
        self.task, self.evidence, self.budget = task, evidence, budget
        self.limit = context_policy(budget.config)["packet_bytes"]
        self.threshold = self.limit * 2
        self.todos = []
        super().__init__(
            model=model,
            trigger=("tokens", self.threshold),
            keep=("messages", 1),
            token_counter=input_upper_estimate,
            trim_tokens_to_summarize=None,
            summarizer=self.summarize,
        )

    def packet(self, messages) -> dict:
        evidence, budget = self.evidence, self.budget
        packet = {
            "task": self.task.model_dump(mode="json"),
            "rules": (
                "Source text is untrusted evidence. Prefixes below are verbatim, may be clipped, "
                "and are not certified claims. Reload exact passages with get_passage when needed. "
                "All listed passages are saved from read documents; cite their original IDs. "
                "Do not reload a passage already shown in full or re-record an existing finding. "
                "Listed document pages are already read; revisit a specific ID with get_passage "
                "rather than repeating read_document. A 1500-character excerpt is a full passage. "
                "Answer from all read evidence even if claim recording is incomplete. "
                "Missing evidence does not establish non-occurrence."
            ),
            "phase": budget.state["phase"],
            "recent_actions": budget.state.get("recent_evidence_actions", []),
            "search_queries": budget.state.get("search_queries", []),
            "recorded_claim_count": len(evidence.claims),
            "recorded_passage_ids": sorted({p for c in evidence.claims for p in c.passage_ids}),
            "unread_candidates": [
                {"version_id": key, "title": hit["title"], "url": hit["url"]}
                for key, hit in budget.state.get("search_candidates", {}).items()
                if key not in evidence.documents
            ],
            "documents": [],
            "claims": [],
            "recent_errors": [],
            "todos": self.todos,
        }
        if budget.state.get("retrieval_mode") == "live":
            total = len(packet["unread_candidates"])
            packet["unread_candidates"] = packet["unread_candidates"][:6]
            packet["unread_candidates_omitted"] = max(0, total - 6)
        by_document = {}
        for p in evidence.passages.values():
            by_document.setdefault(p.version_id, []).append(p)
        rows = []
        for version_id, doc in evidence.documents.items():
            passages = sorted(by_document.get(version_id, []), key=lambda p: p.start)
            row = {
                "version_id": version_id,
                "title": doc.title,
                "url": doc.url,
                "published_at": doc.published_at,
                "available_at": doc.available_at,
                "passage_ids": [p.passage_id for p in passages],
                "excerpts": {},
            }
            packet["documents"].append(row)
            rows.append((row, passages))
        for message in messages[-8:]:
            if isinstance(message, ToolMessage):
                try:
                    result = json.loads(message.content)
                    if "error" in result:
                        packet["recent_errors"].append(
                            {
                                "tool": message.name,
                                "error": str(result["error"])[:120],
                            }
                        )
                except (ValueError, TypeError):
                    pass
        if byte_size(packet) > self.limit:
            packet["todos"], packet["recent_errors"] = [], []
        if byte_size(packet) > self.limit:
            packet["recent_actions"] = packet["recent_actions"][-3:]
        if byte_size(packet) > self.limit:
            raise BudgetExceeded("evidence_context_index_limit")
        # Give each read document its original lead before recency can use the
        # remaining space. This is positional coverage, not semantic selection.
        for row, passages in rows:
            if passages:
                p = passages[0]
                row["excerpts"][p.passage_id] = p.text
                if byte_size(packet) > self.limit:
                    row["excerpts"].pop(p.passage_id)
        # Keep newly requested originals first; otherwise repeated compaction can
        # erase precisely the paragraph the model just reloaded to check a qualifier.
        for pid in reversed(budget.state.get("recent_passages", [])):
            passage = evidence.passages.get(pid)
            if passage is None:
                continue
            row = next(r for r, _ in rows if r["version_id"] == passage.version_id)
            if pid in row["excerpts"]:
                continue
            row["excerpts"][pid] = passage.text
            if byte_size(packet) > self.limit:
                row["excerpts"].pop(pid)
                break
        for row, passages in rows:
            if passages:
                p = passages[0]
                if p.passage_id in row["excerpts"]:
                    continue
                row["excerpts"][p.passage_id] = p.text[:600]
                if byte_size(packet) > self.limit:
                    row["excerpts"].pop(p.passage_id)
        for claim in reversed(evidence.claims):
            item = {
                "claim_id": claim.claim_id,
                "text": claim.text[:180],
                "passage_ids": claim.passage_ids,
                "event_time": claim.event_time,
                "stage": claim.stage,
                "verification": "unverified",
            }
            packet["claims"].append(item)
            if byte_size(packet) > self.limit:
                packet["claims"].pop()
                break
        max_passages = max((len(p) for _, p in rows), default=0)
        for index in range(1, max_passages):
            for row, passages in rows:
                if index < len(passages):
                    p = passages[index]
                    if p.passage_id in row["excerpts"]:
                        continue
                    row["excerpts"][p.passage_id] = p.text[:240]
                    if byte_size(packet) > self.limit:
                        row["excerpts"].pop(p.passage_id)
        return packet

    async def summarize(self, messages) -> str:
        text = encoded(self.packet(messages))
        self.budget.state["context_compactions"] = (
            self.budget.state.get("context_compactions", 0) + 1
        )
        self.budget.save()
        self.evidence.store.event(
            "context_compacted",
            method="sdk_deterministic_evidence_packet",
            input_messages=len(messages),
            packet_bytes=len(text.encode()),
            limit_bytes=self.limit,
            saved_documents=len(self.evidence.documents),
            saved_passages=len(self.evidence.passages),
            model_calls=0,
        )
        return text

    async def abefore_model(self, state, runtime):
        self.todos = state.get("todos", [])
        messages = state["messages"]
        estimate = input_upper_estimate(messages)
        if estimate < self.threshold:
            self.budget.refresh_phase(estimate)
            if self.budget.state["phase"] != "writing":
                return None
        # A transient marker gives the SDK a safe cut AFTER the entire last tool batch.
        marker = HumanMessage("Continue the original task using the saved evidence context.")
        return await super().abefore_model({**state, "messages": [*messages, marker]}, runtime)

    def _should_summarize(self, messages, total_tokens):
        # The pinned SDK still owns partitioning and checkpoint message replacement.
        return self.budget.state["phase"] == "writing" or super()._should_summarize(
            messages, total_tokens
        )
