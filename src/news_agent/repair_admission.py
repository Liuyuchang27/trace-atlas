"""C3 admission only: one persistent repair slot, never a search or budget extension."""

from __future__ import annotations

from .answer_contract import cutoff_for
from .question_state import digest
from .storage import read_json


def policy():
    return {
        "version": "bounded-gap-repair-admission-v1",
        "automatic_execution": False,
        "max_attempts": 1,
        "scope": "main_typed_completion_gap_only",
        "min_research_model_calls": 3,
        "min_research_tool_calls": 3,
        "min_research_seconds": 120,
        "budget": "original_research_allocation_preserve_writing_reserves",
        "conflicts": "no_automatic_arbitration",
        "reservation": "scheduling_slot_only_no_fake_model_or_token_usage",
    }


class RepairAdmission:
    """An opt-in future executor gate. Construction and check are read-only.

    The caller must supply the conservative *actual* next-request input estimate including
    tool schemas. Each actual request must still pass the existing Budget boundaries. This
    check cannot guarantee provider latency, future context growth, credit use or fact quality.
    A reserved attempt stays consumed after cancellation/crash. There is no executor here.
    """

    def __init__(self, questions, budget):
        self.questions, self.budget = questions, budget
        self.path = budget.store.root / "repair_state.json"
        self.data = (
            read_json(self.path)
            if self.path.exists()
            else {
                "version": policy()["version"],
                "task_sha256": questions.data["task_sha256"],
                "attempt": None,
            }
        )
        if self.path.exists():
            if self.data.get("sha256") != digest(
                {k: v for k, v in self.data.items() if k != "sha256"}
            ):
                raise ValueError("repair state checksum mismatch")
            if (
                self.data["version"] != policy()["version"]
                or self.data["task_sha256"] != questions.data["task_sha256"]
            ):
                raise ValueError("repair state policy/task changed")

    def _save(self):
        self.data["sha256"] = digest({k: v for k, v in self.data.items() if k != "sha256"})
        self.budget.store.save("repair_state.json", self.data)

    def check(self, input_estimate, question_id="main", *, _checking_reserved=False):
        if type(input_estimate) is not int or input_estimate <= 0:
            raise ValueError("provide actual positive conservative request input estimate")
        qid = self.questions.resolve(question_id)
        q = self.questions.data["questions"][qid]
        s, c, p = self.budget.state, self.budget.config, policy()
        result = {
            "admitted": False,
            "action": "stop_repair",
            "question_id": qid,
            "question_state_sha256": self.questions.data["sha256"],
            "evidence_revision": q["evidence_revision"],
            "input_estimate": input_estimate,
            "automatic_execution": False,
        }

        def no(reason, action="stop_repair"):
            return {**result, "reason": reason, "action": action}

        if self.data["attempt"] is not None and not _checking_reserved:
            return no("repair_attempt_already_consumed")
        if qid != "main" or q["kind"] != "completion" or not q["requires_date"]:
            return no("outside_typed_main_completion_scope")
        if not cutoff_for(self.questions.task)["date"]:
            return no("missing_cutoff_requires_user_scope_not_search")
        if self.questions.data["halt_reason"]:
            return no("run_already_stopped")
        if q["status"] == "answered_provisionally":
            return no("already_provisionally_covered")
        if q["status"] == "conflicted":
            return no("conflict_requires_explicit_review")
        if q["status"] == "unavailable":
            return no("source_unavailability_not_an_automatic_retry")
        if q["status"] in {"pending_review", "evidence_received"}:
            return no("assess_existing_originals_before_search", "assess_saved")
        if q["status"] == "awaiting_evidence":
            return no("initial_research_not_a_repair", "initial_research")
        if q["status"] != "partial_support":
            return no("no_recoverable_typed_gap")
        if s["phase"] != "research":
            return no("research_closed_preserve_writing_reserve")
        if s.get("pending") or s.get("retrieval_pending"):
            return no("request_in_flight")
        if s.get("search_rounds", 0) >= 6:
            return no("search_round_limit")
        if (
            c.max_model_calls is not None
            and c.max_model_calls - c.reserve_model_calls - s["model_calls"]
            < p["min_research_model_calls"]
        ):
            return no("insufficient_research_model_calls")
        if (
            c.max_tool_calls is not None
            and c.max_tool_calls - c.reserve_tool_calls - s["tool_calls"]
            < p["min_research_tool_calls"]
        ):
            return no("insufficient_research_tool_calls")
        token_floor = p["min_research_model_calls"] * (input_estimate + c.max_output_tokens)
        if (
            c.max_tokens is not None
            and c.max_tokens - c.reserve_tokens - self.budget.used_tokens() < token_floor
        ):
            return no("insufficient_research_tokens")
        if self.budget.remaining_time() <= c.reserve_seconds + p["min_research_seconds"]:
            return no("insufficient_research_time")
        return {
            **result,
            "admitted": True,
            "action": "propose_one_targeted_search",
            "reason": "typed_gap_with_original_research_capacity",
            "minimum_tokens": token_floor,
        }

    def reserve(self, input_estimate, question_id="main"):
        # The harness owns a single worker per run. Re-read within its budget lock so
        # another in-process gate handle cannot reserve against an old empty snapshot.
        with self.budget._lock:
            self.data = RepairAdmission(self.questions, self.budget).data
            decision = self.check(input_estimate, question_id)
            if not decision["admitted"]:
                return decision
            self.data["attempt"] = {**decision, "status": "reserved_not_executed"}
            self._save()  # write ahead; a crash does not reset the scheduling slot
            self.budget.store.event("repair_slot_reserved", **self.data["attempt"])
            return self.data["attempt"]

    def validate_for_dispatch(self):
        """Future executor must recheck immediately before dispatch; never dispatch here."""
        self.data = RepairAdmission(self.questions, self.budget).data
        attempt = self.data["attempt"]
        if attempt is None or attempt["status"] != "reserved_not_executed":
            return {"admitted": False, "reason": "no_reserved_attempt"}
        if self.questions.data["sha256"] != attempt["question_state_sha256"]:
            return {"admitted": False, "reason": "evidence_or_question_state_changed"}
        return self.check(
            attempt["input_estimate"], attempt["question_id"], _checking_reserved=True
        )

    def cancel(self, reason):
        with self.budget._lock:
            self.data = RepairAdmission(self.questions, self.budget).data
            if self.data["attempt"] is not None:
                self.data["attempt"].update(
                    status="cancelled_without_execution", reason=reason[:200]
                )
                self._save()
        # Intentionally no refund/reset. The budget ledger is untouched.
