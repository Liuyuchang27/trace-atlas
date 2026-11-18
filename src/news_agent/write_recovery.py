"""One explicit, metered model decision to recover a rejected node write."""

import copy
import json
from typing import Annotated, Literal

from langchain_core.tools import tool
from pydantic import Field

from .event_graph import BoundFinding, Finding, UndatedBoundFinding, digest
from .storage import read_json


class EventWriteRecovery:
    def __init__(self, graph, budget):
        self.graph, self.budget, self.store = graph, budget, graph.store
        self.path = self.store.root / "event_write_recovery.json"
        self.data = (
            read_json(self.path)
            if self.path.exists()
            else {
                "version": "node-write-recovery-v1",
                "task_sha256": graph.data["task_sha256"],
                "proposal": None,
            }
        )
        if self.path.exists() and self.data.get("sha256") != digest(
            {k: v for k, v in self.data.items() if k != "sha256"}
        ):
            raise ValueError("write recovery checksum mismatch")
        if (
            self.data["version"] != "node-write-recovery-v1"
            or self.data["task_sha256"] != graph.data["task_sha256"]
        ):
            raise ValueError("write recovery policy/task changed")
        p = self.proposal
        if (
            p
            and p["status"] == "dispatching"
            and any(
                f["finding_id"] == p["finding_id"]
                for f in graph.data["nodes"][p["node_id"]]["findings"]
            )
        ):
            p["status"] = "recovered"  # commit survived; do not ask/pay again
        self.save()

    @property
    def proposal(self):
        return self.data["proposal"]

    def save(self):
        self.data["sha256"] = digest({k: v for k, v in self.data.items() if k != "sha256"})
        self.store.save("event_write_recovery.json", self.data)

    def pending(self):
        return bool(self.proposal and self.proposal["status"] == "pending")

    def waiting(self):
        return bool(
            self.proposal and self.proposal["status"] in {"pending", "offered", "dispatching"}
        )

    def capture(self, call, payload):
        stale = payload.get("error") == "stale node revision"
        date_error = bool(payload.get("date_correction")) and (
            payload.get("error")
            in {
                "event/applicability date must occur in original quotes, not metadata",
                "year absent from original",
            }
            or str(payload.get("error", "")).startswith("date provenance:")
        )
        if call["name"] not in {"append_event_finding", "append_undated_finding"} or not (
            stale or date_error
        ):
            return
        if self.proposal:
            return  # cached rejected results and later errors cannot create another opportunity
        args = call["args"]
        try:
            node = self.graph.data["nodes"][args["node_id"]]
            undated = call["name"] == "append_undated_finding"
            finding = (UndatedBoundFinding if undated else BoundFinding).model_validate(
                args["finding"]
            )
            resolved = (
                self.graph.resolve_undated_selected(finding)
                if undated
                else self.graph.resolve_selected(finding)
            )
            self.graph._references(resolved.originals)
            if not resolved.originals:
                return
            if stale:
                self.graph._time_in_original(resolved.event_time, resolved.originals)
                self.graph._time_in_original(resolved.valid_time, resolved.originals)
        except (ValueError, KeyError, TypeError):
            return
        self.data["proposal"] = {
            "recovery_id": "wr-" + digest([call["id"], args])[:16],
            "source_action_id": call["id"],
            "node_id": node["node_id"],
            "finding": copy.deepcopy(finding.model_dump(mode="json")),
            "source_tool": call["name"],
            "kind": "date_support" if date_error else "stale_revision",
            "rejected_revision": args["expected_revision"],
            "finding_id": "ef-" + digest([node["node_id"], resolved.model_dump()])[:16],
            "status": "pending",
            "model_opportunities": 0,
        }
        self.save()
        self.store.event("event_write_recovery_queued", **self.summary())

    def summary(self):
        p = self.proposal
        if not p:
            return None
        return {
            **{
                k: p[k]
                for k in [
                    "recovery_id",
                    "node_id",
                    "status",
                    "source_action_id",
                    "model_opportunities",
                ]
            },
            "current_revision": len(self.graph.data["nodes"][p["node_id"]]["findings"]),
            **({"error": p["error"]} if p.get("error") else {}),
        }

    def references(self):
        p = self.proposal
        return self._resolved().originals if p else []

    def _resolved(self):
        p = self.proposal
        if p.get("resolution_finding"):
            return Finding.model_validate(p["resolution_finding"])
        if p.get("source_tool") == "append_undated_finding":
            return self.graph.resolve_undated_selected(
                UndatedBoundFinding.model_validate(p["finding"])
            )
        return self.graph.resolve_selected(BoundFinding.model_validate(p["finding"]))

    def offer(self):
        if not self.pending():
            raise ValueError("write recovery opportunity already consumed")
        self.proposal.update(status="offered", model_opportunities=1)
        self.save()  # before a model request; interruption does not grant another paid turn
        self.store.event("event_write_recovery_offered", **self.summary())
        instruction = (
            "One decision only. Review this unchanged rejected finding "
            "and current revision. Use resolve_event_write to retry it "
            "as a NEW action at that explicit revision, or skip with a reason. "
            "The old call remains rejected. No searches or extra budget."
        )
        if self.proposal.get("kind") == "date_support":
            instruction = (
                "The originals were read but the submitted full date was unsupported. "
                "Do not repeat it or infer a year. Choose bind_unknown with explicit "
                "expected_revision, a revised assertion keeping only supported raw month/day, "
                "exact date_text from these same originals if any date is printed; otherwise omit "
                "date_text. Give a reason explaining the date gap. "
                "Both event/applicability times stay unknown ONLY after this explicit decision. "
                "Or choose skip. One shared recovery opportunity, no extra budget; "
                "the original rejected call is unchanged."
            )
        return "EVENT_WRITE_RECOVERY:\n" + json.dumps(
            {
                **self.summary(),
                "rejected_finding": self.proposal["finding"],
                "instruction": instruction,
            },
            ensure_ascii=False,
        )

    def exhaust(self, reason):
        if self.pending():
            self.proposal.update(status="exhausted", error=reason)
            self.save()
            self.store.event("event_write_recovery_exhausted", **self.summary())

    def abandon_unfinished(self):
        if self.proposal and self.proposal["status"] in {"offered", "dispatching"}:
            self.proposal.update(
                status="interrupted", error="prior offered decision did not complete"
            )
            self.save()

    def resolve(
        self, recovery_id, decision, expected_revision, reason, assertion=None, date_text=None
    ):
        p = self.proposal
        if not p or p["recovery_id"] != recovery_id:
            raise ValueError("choose the recorded recovery_id")
        if p["status"] == "recovered":
            return {"already_recovered": True, "write_recovery": self.summary()}
        action = self.budget.state.get("active_tool_action_id")
        if p["status"] != "offered" and not (
            p["status"] == "dispatching" and p.get("retry_action_id") == action
        ):
            raise ValueError("no further write recovery attempt is allowed")
        if not reason.strip() or len(reason) > 240:
            p.update(status="failed", error="give a concise decision reason")
            self.save()
            return {"error": p["error"], "write_recovery": self.summary()}
        if decision == "skip":
            p.update(status="skipped", reason=reason)
            self.save()
            return {"write_recovery": self.summary(), "unresolved": True}
        date_error = p.get("kind") == "date_support"
        if decision != ("bind_unknown" if date_error else "retry"):
            raise ValueError(
                "choose bind_unknown or skip" if date_error else "choose retry or skip"
            )
        p.update(status="dispatching", retry_action_id=action, reason=reason)
        self.save()
        try:
            if type(expected_revision) is not int:
                raise ValueError("retry requires the explicit current node revision")
            if date_error:
                # The model authors this replacement explicitly. Never strip dates or
                # repair assertion text on its behalf, including unsupported years.
                schema = UndatedBoundFinding if date_text is not None else BoundFinding
                selected = schema.model_validate(
                    {
                        **{
                            k: v
                            for k, v in p["finding"].items()
                            if k not in {"event_time", "valid_time", "uncertainty"}
                        },
                        "assertion": assertion,
                        **({"date_text": date_text} if date_text is not None else {}),
                        "uncertainty": reason,
                        # Removing an unsupported time basis cannot establish the full
                        # original question/cutoff. Keep its coverage obligation open.
                        "answers_node_question": False,
                        "answers_question_part": False,
                    }
                )
                resolved = (
                    self.graph.resolve_undated_selected(selected)
                    if date_text is not None
                    else self.graph.resolve_selected(selected)
                )
                p["resolution_finding"] = resolved.model_dump(mode="json")
                p["date_resolution"] = selected.model_dump(mode="json")
                p["finding_id"] = "ef-" + digest([p["node_id"], resolved.model_dump()])[:16]
                self.save()  # explicit decision survives a crash after graph commit
            self.graph.append(p["node_id"], self._resolved(), expected_revision)
        except (ValueError, KeyError) as exc:
            p.update(status="failed", error=str(exc))
            self.save()
            self.store.event("event_write_recovery_failed", **self.summary())
            return {"error": str(exc), "write_recovery": self.summary()}
        p["status"] = "recovered"
        self.save()
        self.store.event("event_write_recovery_committed", **self.summary())
        return {"write_recovery": self.summary(), "finding_id": p["finding_id"]}

    def tools(self):
        @tool
        async def resolve_event_write(
            recovery_id: str,
            decision: Literal["retry", "skip", "bind_unknown"],
            reason: Annotated[str, Field(min_length=1, max_length=240)],
            expected_revision: int | None = None,
            assertion: str | None = None,
            date_text: str | None = None,
        ) -> str:
            """One retry/skip; reason <=240 characters. Date repair needs a revised assertion.

            bind_unknown uses recorded originals, keeps both dates unknown and requires
            a revised assertion with no unsupported year. Supply exact date_text only when
            present in the original; omit it for fully undated originals. No automatic repair.
            """
            try:
                return json.dumps(
                    self.resolve(
                        recovery_id, decision, expected_revision, reason, assertion, date_text
                    ),
                    ensure_ascii=False,
                )
            except (ValueError, KeyError) as exc:
                return json.dumps(
                    {"error": str(exc), "write_recovery": self.summary()}, ensure_ascii=False
                )

        return [resolve_event_write]
