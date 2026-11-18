"""Public action summaries for the test UI; never hidden model reasoning."""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import Field, field_validator

from .schemas import StrictModel

PROGRESS_PROMPT = """
PUBLIC INVESTIGATION MAP:
Use publish_plan to give the user a short actionable plan before research. Update the relevant
step before changing direction, when evidence is insufficient, when seeking an original source,
or when expanding/focusing an event time range. Keep summaries brief: the action, the observable
gap and the next evidence needed. Do not disclose private reasoning or fabricate actions.
Use stable step_id values and parent_id to link a follow-up to its gap. Mark status accurately;
completed is your reported progress, not certification. Include read passage IDs when relevant.
Mark the user's main status/completion evidence goal with answer_role="main_status" and start it
before background. Use answer_role="date_anchor" for the requested cutoff/decision date evidence.
Keep an unstarted goal planned, and update it when its actual query/read occurs. Goal roles are
your proposed evidence needs, not verified conclusions. Batch short updates with useful actions.
Publishing a plan does not execute searches: use the real tools afterwards. Preserve the same
research budget and prioritize the user's question. Do not invent a temporal backtrack if none
occurred. Finish with a concise report and dated timeline with original passage citations.
Each search requires purpose and intent: state the short observable evidence need, not private
reasoning. Use based_on for relevant already-read passages and step_id when an existing plan fits.
If a selected page starts with navigation/cookies, use find_saved_text with your own keywords to
locate relevant exact passages in that already-saved page instead of reading all navigation pages.
"""


def progress_policy():
    return {
        "version": "public-plan-v3",
        "max_steps": 24,
        "private_reasoning": False,
        "search_intent": "required_public_evidence_need_no_extra_model_request",
        "saved_document_search": "model_chosen_keywords_exact_saved_passages",
    }


class SearchIntent(StrictModel):
    query: str = Field(min_length=1, max_length=600)
    purpose: str = Field(min_length=1, max_length=240)
    intent: Literal["initial", "gap", "source_trace", "temporal", "followup"]
    limit: int = Field(default=5, ge=1, le=10)
    step_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,48}$")
    based_on: list[str] = Field(default_factory=list, max_length=6)
    time_start: str | None = Field(default=None, max_length=40)
    time_end: str | None = Field(default=None, max_length=40)
    include_domains: list[str] | None = Field(default=None, max_length=8)


def visual_search_tool(original, evidence, budget):
    @tool("search", args_schema=SearchIntent)
    async def search(
        query: str,
        purpose: str,
        intent: str,
        config: RunnableConfig,
        limit: int = 5,
        step_id: str | None = None,
        based_on: list[str] | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        include_domains: list[str] | None = None,
    ) -> str:
        """Search with a short public evidence goal. State initial/gap/source/time/follow-up intent.

        purpose describes needed evidence, not private reasoning. Optional based_on must cite
        already-read passage IDs. step_id must name your existing published plan, if supplied.
        """
        from .storage import read_json

        if set(based_on or []) - set(budget.state.get("delivered_passages", [])):
            return json.dumps({"error": "search intent must cite read passages"})
        path = evidence.store.root / "public_plan.json"
        requested_step = step_id
        warning = None
        if step_id and (not path.exists() or step_id not in read_json(path)):
            warning = "unknown_plan_id_search_executed_without_plan_link"
            step_id = None  # optional UI metadata must not block valid research
        evidence.store.event(
            "search_intent",
            query=query,
            purpose=purpose,
            intent=intent,
            step_id=step_id,
            based_on=based_on or [],
            time_start=time_start,
            time_end=time_end,
            provenance="model_reported_action_summary",
            requested_step_id=requested_step,
            plan_link_warning=warning,
            include_domains=include_domains,
        )
        arguments = {"query": query, "limit": limit}
        if include_domains:
            if "include_domains" not in original.args:
                return json.dumps({"error": "domain filtering is available only for live search"})
            arguments["include_domains"] = include_domains
        result = await original.ainvoke(arguments, config=config)
        state = getattr(evidence, "question_state", None)
        if state:
            state.link_candidates(requested_step, json.loads(result).get("results", []))
        if warning:
            result = json.dumps({**json.loads(result), "plan_link_warning": warning})
        return result

    return search


def document_locator(evidence, budget):
    @tool
    async def find_saved_text(version_id: str, query: str, limit: int = 4) -> str:
        """Find your keywords inside an already-selected, extracted document, avoiding navigation.

        No web request. Returns exact original passages, ranked by your phrase/keyword matches.
        A search snippet or an unextracted candidate cannot be opened by this tool.
        """
        from .budget import BudgetExceeded
        from .evidence_chain import expose_passages, rank_passages, with_date_context
        from .harness import remember_passages

        try:
            budget.before_tool("read")
            if version_id not in evidence.documents or not query.strip():
                return json.dumps({"error": "select and extract the page first; provide keywords"})
            matches = [
                r
                for r in rank_passages(
                    (p for p in evidence.passages.values() if p.version_id == version_id), query
                )
                if r["matched_terms"]
            ]
            selected = [r["passage"] for r in matches[: max(1, min(limit, 6))]]
            core = budget.state.get("evidence_chain_version")
            if core:
                selected = with_date_context(
                    (p for p in evidence.passages.values() if p.version_id == version_id), selected
                )
            ids = [p.passage_id for p in selected]
            if core and ids and set(ids) <= set(budget.state.get("delivered_passages", [])):
                return json.dumps(
                    {
                        "version_id": version_id,
                        "already_delivered": True,
                        "passage_ids": ids,
                        "query": query,
                        "instruction": (
                            "Use saved focused originals. Reload a qualifier with get_passage "
                            "only if exposed; otherwise follow a source link, bind evidence, "
                            "or close_gap_round with the unresolved need."
                            if hasattr(evidence, "adaptive")
                            else (
                                "Use saved focused originals; "
                                "get_passage reloads a specific qualifier."
                            )
                        ),
                    }
                )
            if core:
                expose_passages(
                    evidence,
                    budget,
                    selected,
                    method="find_saved_text",
                    query=query,
                    scores={r["passage"].passage_id: r for r in matches},
                )
            else:
                remember_passages(budget, ids)
            doc = evidence.documents[version_id]
            return json.dumps(
                {
                    "version_id": version_id,
                    "url": doc.url,
                    "query": query,
                    "matching_passages": len(matches),
                    "passages": [p.model_dump() for p in selected],
                    "method": (
                        "weighted keywords in immutable saved content; not semantic verification"
                    ),
                }
            )
        except BudgetExceeded as exc:
            return json.dumps({"error": str(exc)})

    return find_saved_text


class PublicStep(StrictModel):
    step_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,48}$")
    title: str = Field(min_length=1, max_length=120)
    kind: Literal["plan", "gap", "followup", "temporal", "source_check", "synthesis"] = "plan"
    status: Literal["planned", "in_progress", "completed", "blocked"] = "planned"
    answer_role: Literal["main_status", "date_anchor", "supporting"] = "supporting"
    evidence_kind: (
        Literal["completion", "approval", "consultation", "announcement", "any"] | None
    ) = None
    requires_date: bool = False
    reason: str = Field(default="", max_length=240)
    parent_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,48}$")
    passage_ids: list[str] = Field(default_factory=list, max_length=6)
    time_start: str | None = Field(default=None, max_length=40)
    time_end: str | None = Field(default=None, max_length=40)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_pending(cls, value):
        return "planned" if value == "pending" else value


def plan_tool(evidence, budget):
    @tool
    async def publish_plan(steps: list[PublicStep]) -> str:
        """Publish 1-8 concise public plan/gap updates and link follow-ups to parent steps.

        These are user-facing action summaries, not private reasoning or executed web requests.
        Cite only read passages. A temporal step may specify the investigated date range.
        """
        if not 1 <= len(steps) <= 8:
            return json.dumps({"error": "publish 1-8 steps at once"})
        path = evidence.store.root / "public_plan.json"
        from .storage import read_json

        current = read_json(path) if path.exists() else {}
        state = getattr(evidence, "question_state", None)
        updates = {
            s.step_id: (
                {**current[s.step_id], **s.model_dump(exclude_unset=True)}
                if state and s.step_id in current
                else s.model_dump()
            )
            for s in steps
        }
        updated = {**current, **updates}
        if len(updated) > progress_policy()["max_steps"]:
            return json.dumps({"error": "public plan step limit reached"})
        delivered = set(budget.state.get("delivered_passages", []))
        for step in steps:
            if step.parent_id and step.parent_id not in updated:
                return json.dumps({"error": "parent step unavailable"})
            if set(step.passage_ids) - delivered:
                return json.dumps({"error": "cite only passages actually read"})
        # Validate the whole proposed state before mutation, including indirect cycles.
        for identifier in updated:
            seen, cursor = set(), identifier
            while cursor:
                if cursor in seen:
                    return json.dumps({"error": "plan links must not form a cycle"})
                seen.add(cursor)
                cursor = updated[cursor].get("parent_id")
        if state:
            try:
                state.register_plan(list(updates.values()))
            except ValueError as exc:
                return json.dumps({"error": str(exc)})
        evidence.store.save("public_plan.json", updated)
        evidence.store.event(
            "public_plan_updated",
            steps=list(updates.values()) if state else [s.model_dump() for s in steps],
            provenance="model_reported_action_summary",
        )
        active = [s.step_id for s in steps if s.status == "in_progress"]
        if active:
            budget.state["public_step_id"] = active[-1]
            budget.save()
        return json.dumps(
            {
                "saved_steps": [s.step_id for s in steps],
                "model_reported": True,
                **({"question_state": state.summary()} if state else {}),
            }
        )

    return publish_plan
