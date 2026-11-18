"""C2 online evidence needs. Status means model-assessed coverage, never verified truth."""

from __future__ import annotations

import copy
import hashlib
import json

from langchain_core.tools import tool

from .answer_contract import Observation, applies, cutoff_for, validate_observation
from .storage import read_json


def policy():
    return {
        "version": "question-evidence-state-v1.1",
        "max_questions": 12,
        "max_bindings_per_question": 12,
        "context_summary_questions": 4,
        "completion": "provisional_coverage_not_fact_verification",
        "updates": "delivered_originals_only_explicit_question_and_seen_revision",
        "conflict": "reopen_never_close_by_plan_status_or_repeat_support",
        "model_calls": 0,
    }


PROMPT = """
ONLINE EVIDENCE STATE:
The main user question has question_id="main". publish_plan may define additional evidence
questions using evidence_kind (completion/approval/consultation/announcement/any) and
requires_date. Main-status plan IDs alias main. Plan completed never closes an evidence gap.
Search with the matching step_id; reads attach delivered originals to that question.
After reading a decisive original, use update_question_evidence with its exact Observation
(kind, short exact subject, target_event, occurrence, event_date, passage_id, quote) and the
question's current seen_revision from the returned question_state. Batch independent tool calls.
Copy subject verbatim from quote in the source language (e.g. "Acme" in an English quote),
never translate it or replace it with a summary. Keep the question text in the user's language.
Approval/plans do not answer a completion need. stance="contradicts" flags a disputed record;
never hide a conflict by repeating another support. New linked text reopens review; an older
seen_revision cannot clear it. Query update_question_evidence(question_id) for current state.
If unavailable, report unavailable_reason; this leaves an open gap, not a negative fact.
All coverage is provisional/model-assessed. Preserve the original total budget.
An "any" requirement only collects material; define a new explicit kind before treating it as
covered. A question ID/alias cannot silently change its criterion or scope.
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class QuestionState:
    def __init__(self, task, evidence, budget):
        self.task, self.evidence, self.budget = task, evidence, budget
        self.path = evidence.store.root / "question_state.json"
        binding = digest(task.model_dump(mode="json"))
        if self.path.exists():
            self.data = read_json(self.path)
            if self.data.get("sha256") != digest(
                {k: v for k, v in self.data.items() if k != "sha256"}
            ):
                raise ValueError("question state hash mismatch")
            if self.data["version"] != policy()["version"] or self.data["task_sha256"] != binding:
                raise ValueError("question state policy/task changed")
            for link in self.data["bindings"].values():
                self._validate(Observation.model_validate(link["observation"]))
            journal = evidence.store.root / "trajectory.jsonl"
            logged = (
                {e.get("transition_id") for e in map(json.loads, journal.read_text().splitlines())}
                if journal.exists()
                else set()
            )
            for transition in self.data.get("transitions", []):
                if transition["transition_id"] not in logged:
                    evidence.store.event(
                        "question_state_changed", **transition, replayed_after_checkpoint=True
                    )
        else:
            self.data = {
                "version": policy()["version"],
                "task_sha256": binding,
                "questions": {},
                "aliases": {},
                "bindings": {},
                "candidates": {},
                "halt_reason": None,
                "transitions": [],
            }
            self.data["questions"]["main"] = self._new(
                "main",
                task.question.split("\n\nRun observation context:", 1)[0],
                "completion" if applies(task) else "any",
                applies(task),
            )
            self._refresh("main")
            self._save()
            evidence.store.event("question_defined", question_id="main", provenance="user_request")

    def _new(self, qid, text, kind, requires_date):
        return {
            "question_id": qid,
            "text": text,
            "kind": kind,
            "requires_date": requires_date,
            "received_passage_ids": [],
            "evidence_revision": 0,
            "reviewed_revision": -1,
            "binding_ids": [],
            "unavailable_reason": None,
            "status": "awaiting_evidence",
            "gap_status": "open_no_evidence",
            "missing": [],
            "provenance": "model_assessed_unverified",
        }

    def _save(self):
        self.data["sha256"] = digest({k: v for k, v in self.data.items() if k != "sha256"})
        self.evidence.store.save("question_state.json", self.data)

    def resolve(self, qid):
        qid = self.data["aliases"].get(qid, qid)
        if qid not in self.data["questions"]:
            raise ValueError("unknown evidence question; define it with publish_plan")
        return qid

    def register_plan(self, steps):
        before = copy.deepcopy(self.data)
        added = []
        try:
            for s in steps:
                role = s.get("answer_role", "supporting")
                if role == "main_status":
                    if s["step_id"] in self.data["questions"] and s["step_id"] != "main":
                        raise ValueError("question ID cannot change scope; use a new ID")
                    self.data["aliases"][s["step_id"]] = "main"
                    continue
                if (
                    not s.get("evidence_kind")
                    and role != "date_anchor"
                    and s["kind"] not in {"gap", "temporal", "source_check"}
                ):
                    continue
                qid = s["step_id"]
                if qid in self.data["aliases"] and self.data["aliases"][qid] != qid:
                    raise ValueError("question alias cannot change scope; use a new ID")
                kind = s.get("evidence_kind") or "any"
                needs_date = s.get("requires_date", False) or role == "date_anchor"
                if qid in self.data["questions"]:
                    q = self.data["questions"][qid]
                    if (q["kind"], q["requires_date"]) != (kind, needs_date):
                        raise ValueError("evidence criterion is immutable; use a new question ID")
                else:
                    if len(self.data["questions"]) >= policy()["max_questions"]:
                        raise ValueError("evidence question limit")
                    self.data["questions"][qid] = self._new(qid, s["title"], kind, needs_date)
                    self._refresh(qid)
                    added.append(qid)
            self._save()
            for qid in added:
                self.evidence.store.event(
                    "question_defined",
                    question_id=qid,
                    criterion=self.data["questions"][qid]["kind"],
                    provenance="model_plan",
                    action_id=self.budget.state.get("active_tool_action_id"),
                )
        except ValueError:
            self.data = before
            raise

    def link_candidates(self, step_id, hits):
        try:
            qid = self.resolve(step_id)
        except ValueError:
            return
        for hit in hits:
            refs = self.data["candidates"].setdefault(hit["version_id"], [])
            if qid not in refs:
                refs.append(qid)
        self._save()

    def receive(self, passages):
        by_question = {}
        for p in passages:
            for qid in self.data["candidates"].get(p.version_id, []):
                by_question.setdefault(qid, []).append(p.passage_id)
        for qid, ids in by_question.items():
            q = self.data["questions"][qid]
            fresh = set(ids) - set(q["received_passage_ids"])
            if fresh:
                before = copy.deepcopy(q)
                q["received_passage_ids"] = sorted(set(q["received_passage_ids"]) | fresh)
                q["evidence_revision"] += 1
                self._refresh(qid)
                self._changed(qid, before, "originals_delivered")
        self._save()

    def _validate(self, observation):
        pid = observation.passage_id
        if pid not in self.budget.state.get("delivered_passages", []):
            raise ValueError("question evidence must be tool-delivered original text")
        p = self.evidence.passages[pid]
        packet = {"documents": [{"version_id": p.version_id, "excerpts": {pid: p.text}}]}
        return validate_observation(observation, packet)

    def bind(
        self,
        qid,
        observation,
        seen_revision,
        stance="supports",
        origin="research_tool",
        review=True,
    ):
        qid = self.resolve(qid)
        q = self.data["questions"][qid]
        if type(seen_revision) is not int or seen_revision != q["evidence_revision"]:
            raise ValueError("stale_or_missing_seen_revision; inspect current question_state")
        if stance not in {"supports", "contradicts"}:
            raise ValueError("invalid evidence stance")
        if (
            stance == "contradicts"
            and q["kind"] != "any"
            and (
                observation.kind != q["kind"]
                or (q["kind"] == "completion" and not observation.target_event)
            )
        ):
            raise ValueError("contradiction_must_match_question_event")
        bound = self._validate(observation)
        identity = (
            "qb-"
            + digest(
                [
                    qid,
                    observation.passage_id,
                    observation.kind,
                    observation.target_event,
                    observation.occurrence,
                    observation.event_date,
                    stance,
                ]
            )[:16]
        )
        if (
            identity not in self.data["bindings"]
            and len(q["binding_ids"]) >= policy()["max_bindings_per_question"]
        ):
            raise ValueError("question binding limit")
        before = copy.deepcopy(q)
        pid = observation.passage_id
        if pid not in q["received_passage_ids"]:
            q["received_passage_ids"].append(pid)
            q["evidence_revision"] += 1
        self.data["bindings"].setdefault(
            identity,
            {
                "binding_id": identity,
                "question_id": qid,
                "observation": observation.model_dump(),
                "validated": bound,
                "stance": stance,
                "origin": origin,
                "action_id": self.budget.state.get("active_tool_action_id"),
                "seen_revision": seen_revision,
                "linked_revision": q["evidence_revision"],
                "related_claim_ids": [
                    c.claim_id for c in self.evidence.claims if pid in c.passage_ids
                ],
                "claim_relation": "shared_original_only_not_semantic_certification",
            },
        )
        if identity not in q["binding_ids"]:
            q["binding_ids"].append(identity)
        if review:
            q["reviewed_revision"] = q["evidence_revision"]
        self._refresh(qid)
        self._changed(qid, before, "evidence_bound", binding_id=identity)
        self._save()
        return self.summary(qid)

    def _refresh(self, qid):
        q = self.data["questions"][qid]
        links = [self.data["bindings"][i] for i in q["binding_ids"]]

        def qualifies(link):
            o = link["observation"]
            return (
                q["kind"] != "any"
                and o["kind"] == q["kind"]
                and o["occurrence"] == "observed"
                and (not q["requires_date"] or bool(o["event_date"]))
                and (q["kind"] != "completion" or o["target_event"])
            )

        good = [link for link in links if qualifies(link) and link["stance"] == "supports"]
        if qid == "main" and q["kind"] == "completion" and not cutoff_for(self.task)["date"]:
            good = []
        dates = {link["observation"]["event_date"] for link in good}
        conflict = any(link["stance"] == "contradicts" for link in links) or (
            q["kind"] == "completion" and len(dates) > 1
        )
        if conflict:
            status, gap = "conflicted", "open_conflict"
        elif good and q["reviewed_revision"] < q["evidence_revision"]:
            status, gap = "pending_review", "open_new_material"
        elif good:
            status, gap = "answered_provisionally", "addressed_model_assessed"
        elif q["unavailable_reason"]:
            status, gap = "unavailable", "open_unavailable"
        elif self.data["halt_reason"]:
            status, gap = "unresolved", "open_stopped"
        elif links:
            status, gap = "partial_support", "open_requirements_missing"
        elif q["received_passage_ids"]:
            status, gap = "evidence_received", "open_unassessed"
        else:
            status, gap = "awaiting_evidence", "open_no_evidence"
        q.update(
            status=status,
            gap_status=gap,
            missing=[] if good and not conflict and status == "answered_provisionally" else [gap],
        )

    def _changed(self, qid, before, trigger, **extra):
        q = self.data["questions"][qid]
        if before != q:
            transition = dict(
                question_id=qid,
                trigger=trigger,
                before_status=before["status"],
                after_status=q["status"],
                gap_status=q["gap_status"],
                evidence_revision=q["evidence_revision"],
                action_id=self.budget.state.get("active_tool_action_id"),
                **extra,
            )
            transition["transition_id"] = (
                "qt-"
                + digest([self.data["task_sha256"], len(self.data["transitions"]), transition])[:16]
            )
            self.data["transitions"].append(transition)
            self._save()  # checkpoint the transition before publishing its journal event
            self.evidence.store.event("question_state_changed", **transition)

    def unavailable(self, qid, reason):
        qid = self.resolve(qid)
        q = self.data["questions"][qid]
        before = copy.deepcopy(q)
        q["unavailable_reason"] = {"reason": reason[:200], "provenance": "model_reported"}
        self._refresh(qid)
        self._changed(qid, before, "unavailability_reported")
        self._save()
        return self.summary(qid)

    def finish(self, reason):
        self.data["halt_reason"] = reason
        for qid in self.data["questions"]:
            before = copy.deepcopy(self.data["questions"][qid])
            self._refresh(qid)
            self._changed(qid, before, "run_stopped")
        self._save()

    def summary(self, qid=None):
        selected = (
            [self.data["questions"][self.resolve(qid)]]
            if qid
            else list(self.data["questions"].values())[: policy()["context_summary_questions"]]
        )
        keys = [
            "question_id",
            "text",
            "kind",
            "requires_date",
            "status",
            "gap_status",
            "evidence_revision",
            "reviewed_revision",
            "missing",
        ]
        result = [{k: (q[k][:160] if k == "text" else q[k]) for k in keys} for q in selected]
        for row, q in zip(result, selected, strict=True):
            row["bound_passage_ids"] = sorted(
                {
                    self.data["bindings"][bid]["observation"]["passage_id"]
                    for bid in q["binding_ids"]
                }
            )
        return result

    def pinned_passages(self):
        return {link["observation"]["passage_id"] for link in self.data["bindings"].values()}

    def constraints(self):
        main = self.data["questions"]["main"]
        return {
            "main_status": main["status"],
            "completion_dates": sorted(
                {
                    self.data["bindings"][bid]["observation"]["event_date"]
                    for bid in main["binding_ids"]
                    if self.data["bindings"][bid]["observation"]["kind"] == "completion"
                    and self.data["bindings"][bid]["observation"]["occurrence"] == "observed"
                    and self.data["bindings"][bid]["observation"]["target_event"]
                    and self.data["bindings"][bid]["observation"]["event_date"]
                }
            ),
            "state_version": policy()["version"],
        }

    def ingest_final(self, observations):
        for row in observations:
            if row["kind"] == "completion" and row["target_event"]:
                obj = Observation(**{k: row[k] for k in Observation.model_fields})
                q = self.data["questions"]["main"]
                self.bind(
                    "main",
                    obj,
                    q["evidence_revision"],
                    origin="final_model_binding",
                    review=q["status"] != "pending_review",
                )


def state_tool(state):
    @tool
    async def update_question_evidence(
        question_id: str,
        observation: Observation | None = None,
        seen_revision: int | None = None,
        stance: str = "supports",
        unavailable_reason: str | None = None,
    ) -> str:
        """Inspect a question, or bind an exact original with its current seen_revision.

        Supports/contradicts are model assessments. No manual status/close flag is accepted.
        Plans, approval and publication metadata cannot satisfy a completion-date need.
        """
        try:
            if observation is not None:
                result = state.bind(question_id, observation, seen_revision, stance)
            elif unavailable_reason:
                result = state.unavailable(question_id, unavailable_reason)
            else:
                result = state.summary(question_id)
            return json.dumps({"question_state": result})
        except (ValueError, KeyError) as exc:
            result = {"error": str(exc)}
            if str(exc) == "subject_not_in_quote":
                result["recovery"] = (
                    "Copy a short subject verbatim from the quote in its original language; "
                    "do not translate or paraphrase it. Keep the exact quote/date and inspect "
                    "the current question_state before resubmitting. No state was changed."
                )
            try:
                result["question_state"] = state.summary(question_id)
            except ValueError:
                pass
            return json.dumps(result)

    return update_question_evidence
