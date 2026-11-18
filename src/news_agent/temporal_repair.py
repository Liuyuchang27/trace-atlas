"""Persisted bridge from temporal audit findings to the existing evidence scheduler."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from .event_graph import digest
from .resource_control import execution_limits
from .storage import read_json

FOLLOWUP_GAP_KINDS = {
    "temporal_audit",
    "question_or_explanation_unresolved",
    "conflicting_evidence",
    "missing_original",
}


def policy():
    return {
        "version": "temporal-repair-loop-v3",
        "closure": "independent_reaudit_of_explicit_same_claim_replacement",
        "budget": "existing_ledger_no_reset",
        "no_progress_turns": 6,
        "no_progress_scope": "defer_one_target_then_continue",
        "saved_context": "distinct_lookups_and_direct_search_handoff",
        "repeat_feedback": "durable_rejection_and_hide_repeated_lookup_until_next_round",
        "date_resolution": "reuse_or_append_separately_dated_same_claim_candidate",
        "batch_proposals": "per_target_validation_with_durable_receipt",
        "question_coverage": "separate_audit_never_changed_by_time_link",
        "unknown": "retain_original_no_inferred_year",
    }


class TemporalRepair:
    def __init__(self, adaptive):
        self.a, self.graph, self.budget = adaptive, adaptive.graph, adaptive.budget
        self.evidence, self.store = adaptive.evidence, adaptive.store
        self.path = self.store.root / "temporal_repair.json"
        self.data = (
            read_json(self.path)
            if self.path.exists()
            else {
                "version": policy()["version"],
                "task_sha256": self.graph.data["task_sha256"],
                "targets": {},
                "reviews": [],
                "active": False,
                "audit_requested": False,
            }
        )
        if (
            self.data["version"] not in {"temporal-repair-loop-v2", policy()["version"]}
            or self.data["task_sha256"] != self.graph.data["task_sha256"]
        ):
            raise ValueError("temporal repair policy/task changed")
        if self.path.exists() and self.data.get("sha256") != digest(
            {k: v for k, v in self.data.items() if k != "sha256"}
        ):
            raise ValueError("temporal repair state changed")
        self.data["version"] = policy()["version"]

    def save(self):
        self.data["sha256"] = digest({k: v for k, v in self.data.items() if k != "sha256"})
        self.store.save("temporal_repair.json", self.data)

    def findings(self):
        return {
            r["finding_id"]: (n, r)
            for n in self.graph.data["nodes"].values()
            for r in n["findings"]
        }

    def ingest(self, review):
        if review.get("status") != "reviewed":
            self.stop("audit_unavailable")
            return
        if review["binding"] in self.data["reviews"]:
            return
        self.data["reviews"].append(review["binding"])
        self.data["audit_requested"] = False
        rows = self.findings()
        decisions = {d["finding_id"]: d for d in review["decisions"]}
        for followup in review.get("followups", []):
            fid = followup["finding_id"]
            key = fid + ":" + followup["kind"]
            if key not in self.data["targets"]:
                node, row = rows[fid]
                self.data["targets"][key] = {
                    **followup,
                    "target_id": key,
                    "node_id": node["node_id"],
                    "question_part_id": row["finding"].get("question_part_id"),
                    "importance": node["importance"],
                    "status": "open",
                    "proposals": [],
                    "initial_review": review["binding"],
                }
        for target in self.data["targets"].values():
            if target["status"] == "resolved":
                # Reassess an old resolution against the current audited evidence version.
                target["status"] = "awaiting_review"
                target["previous_resolution"] = {
                    "finding_id": target.pop("resolved_by"),
                    "review_binding": target.pop("review_binding"),
                }
            for candidate in reversed(target["proposals"]):
                decision = decisions.get(candidate["finding_id"], {})
                relation = next(
                    (
                        r
                        for r in decision.get("repairs", [])
                        if r["target_finding_id"] == target["finding_id"]
                    ),
                    {},
                )
                supported = (
                    decision.get("event_verdict") == "supported"
                    and relation.get("event_date_supported")
                    if target["kind"] == "event_date_basis"
                    else relation.get("applicability") in {"supported", "not_applicable"}
                )
                if supported and relation.get("same_event_or_claim"):
                    target.update(
                        status="resolved",
                        resolved_by=candidate["finding_id"],
                        review_binding=review["binding"],
                        resolution_scope="temporal_only",
                        affects_answer_coverage=False,
                        target_applicability=relation.get("applicability"),
                    )
                    self.store.event(
                        "temporal_gap_resolved",
                        target_id=target["target_id"],
                        replacement_finding_id=candidate["finding_id"],
                        review_binding=review["binding"],
                    )
                    break
                if relation:
                    candidate["rejected_by_review"] = {
                        "binding": review["binding"],
                        "reason": relation["reason"],
                    }
            if target["status"] == "awaiting_review":
                target.update(
                    status="open",
                    last_review=review["binding"],
                    last_review_result="replacement_not_supported_or_same_claim_unconfirmed",
                )
        self.save()
        self.a.refresh()

    def gaps(self):
        by_node = {}
        for target in self.data["targets"].values():
            if target["status"] != "resolved":
                by_node[target["node_id"]] = target
        return [
            (nid, "temporal_audit", t["importance"], self.graph.data["nodes"][nid]["title"])
            for nid, t in by_node.items()
        ]

    def repair_links(self, fid):
        rows = self.findings()
        links = [
            {
                "target_finding_id": t["finding_id"],
                "assertion": rows[t["finding_id"]][1]["finding"]["assertion"],
                "occurrence": rows[t["finding_id"]][1]["finding"]["occurrence"],
                "valid_time": rows[t["finding_id"]][1]["finding"]["valid_time"],
                "event_time": rows[t["finding_id"]][1]["finding"]["event_time"],
                "question_part_id": t["question_part_id"],
                "scope": "temporal_only_no_question_coverage_transfer",
                "originals": rows[t["finding_id"]][1]["references"],
                "reason": p["reason"],
            }
            for t in self.data["targets"].values()
            for p in t["proposals"]
            if p["finding_id"] == fid
        ]
        return list({r["target_finding_id"]: r for r in links}.values())

    def summary(self):
        rows = self.findings()
        targets = []
        for t in self.data["targets"].values():
            if t["status"] not in {"open", "awaiting_review"}:
                continue
            node, row = rows[t["finding_id"]]
            targets.append(
                {
                    **{
                        k: t[k]
                        for k in [
                            "target_id",
                            "finding_id",
                            "node_id",
                            "kind",
                            "status",
                            "reason",
                            "passage_ids",
                        ]
                    },
                    "assertion": row["finding"]["assertion"],
                    "question_part_id": t["question_part_id"],
                    "current_revision": len(node["findings"]),
                    "version_ids": list(dict.fromkeys(r["version_id"] for r in row["references"])),
                    "proposals": t["proposals"][-2:],
                    "eligible_saved_candidate_ids": [
                        fid
                        for fid, (candidate_node, candidate) in rows.items()
                        if fid != t["finding_id"]
                        and candidate["references"]
                        and not self.prior_proposal(
                            t, candidate, self.originals_signature(candidate)
                        )
                        and candidate_node["node_id"] == t["node_id"]
                        and (
                            t["kind"] != "event_date_basis"
                            or candidate["finding"]["event_time"]["precision"] != "unknown"
                        )
                    ],
                }
            )
        return {
            "targets": targets,
            "recent_context": self.data.get("recent_context", []),
            "last_action_feedback": self.data.get("last_action_feedback"),
            "temporarily_unavailable_lookups": (self.a.active() or {}).get(
                "blocked_lookup_tools", []
            ),
            "replacement_candidates": [
                {
                    "finding_id": r["finding_id"],
                    "node_id": n["node_id"],
                    "assertion": r["finding"]["assertion"],
                    "event_time": r["finding"]["event_time"],
                    "valid_time": r["finding"]["valid_time"],
                    "question_part_id": r["finding"].get("question_part_id"),
                    "passage_ids": [ref["passage_id"] for ref in r["references"]],
                }
                for n, r in list(rows.values())[-24:]
                if r["references"] and n["node_id"] in {t["node_id"] for t in targets}
            ],
            "audit_requested": self.data["audit_requested"],
            "instruction": (
                "First inspect eligible_saved_candidate_ids: these only match the node, "
                "NOT verified same-claim matches. You MAY propose an existing dated finding "
                "as clarification of the same event WITHOUT editing the undated original. "
                "Different question_part_id is allowed for TIME evidence; this NEVER "
                "transfers answer coverage. A date cannot answer why or prove risk ended. "
                "The separate reviewer must judge its event identity and time support. "
                "Otherwise read saved source context with begin_temporal_context and "
                "find_saved_text/get_passage. Different saved lookups remain available. "
                "If insufficient, search_gap directly from the saved-context round "
                "using your own query, source and window. Reuse an existing finding or append "
                "a NEW source-bound finding in the appropriate question part, then "
                "propose_temporal_resolution for each target it "
                "addresses. Do not rewrite old dates. Call request_temporal_reaudit after "
                "proposals. If attempted evidence cannot establish time, "
                "retain_temporal_unknown with the actual limitation. NEVER backfill the "
                "old source's year from publication/URL/cutoff/another source. A SEPARATE "
                "finding CAN use a date explicitly stated by its OWN cited originals. "
                "No assumed continuous applicability. After reading, submit the eligible "
                "candidate you judge relevant, or explain why none establishes the same "
                "claim; merely rereading or recognizing a date does not submit a proposal. "
                "Review must confirm same event/claim before closure. "
                "A rejected candidate applies only to that target. Skip it, continue other "
                "open fields/questions, or search a different original for that exact field. "
                "Do not repeat a rejected lookup: its returned context is saved above. "
                "Use a different available lookup, search missing context, or explicitly "
                "retain unknown after actual investigation. "
            ),
        }

    def start(self):
        if not any(t["status"] == "open" for t in self.data["targets"].values()):
            return False
        self.a.refresh()
        if not any(
            g["kind"] == "temporal_audit" and g["status"] == "open"
            for g in self.a.data["gaps"].values()
        ):
            self.stop("no_open_productive_audit_gap")
            return False
        if not self.budget.reopen_temporal_research():
            self.stop("research_budget_or_prior_failure")
            return False
        self.data.update(active=True, audit_requested=False, progress=None, stagnant_turns=0)
        self.save()
        self.a.refresh()
        self.store.event("temporal_repair_started", targets=len(self.summary()["targets"]))
        return True

    def stop(self, reason):
        self.data.update(active=False, stop_reason=reason)
        self.save()

    def defer_target(self, target, reason):
        target.update(status="retained_unknown", outcome_reason=reason)
        self.store.event(
            "temporal_target_deferred",
            target_id=target["target_id"],
            node_id=target["node_id"],
            field=target["kind"],
            reason=reason,
        )

    def check_progress(self):
        if not self.data["active"] or self.budget.state["phase"] != "research":
            return
        signature = digest(
            [
                self.graph.data["sha256"],
                self.budget.state.get("delivered_passages", []),
                self.budget.state.get("seen_search_versions", []),
                self.data["targets"],
                [(r["round_id"], r["status"]) for r in self.a.data["rounds"]],
                self.data.get("inspected_context", []),
            ]
        )
        self.data["stagnant_turns"] = (
            self.data.get("stagnant_turns", 0) + 1 if self.data.get("progress") == signature else 0
        )
        self.data["progress"] = signature
        open_targets = [t for t in self.data["targets"].values() if t["status"] == "open"]
        low_yield = next(
            (
                g
                for g in self.a.data["gaps"].values()
                if g["kind"] == "temporal_audit"
                and g["status"] == "open"
                and g["low_yield"]
                >= execution_limits(self.budget.config)["low_yield_rounds_per_gap"]
            ),
            None,
        )
        if low_yield or self.data["stagnant_turns"] >= policy()["no_progress_turns"]:
            if any(t["status"] == "awaiting_review" for t in self.data["targets"].values()):
                return self.request_audit("Review valid pending proposals before further repair")
            feedback = self.data.get("last_action_feedback") or {}
            attempted = set(feedback.get("arguments", {}).get("target_ids", []))
            active = self.a.active()
            node_id = (low_yield or active or {}).get("node_id")
            target = next((t for t in open_targets if t["target_id"] in attempted), None)
            target = target or next((t for t in open_targets if t["node_id"] == node_id), None)
            if target is None and open_targets:
                target = open_targets[0]
            if target:
                self.defer_target(target, "此字段连续操作未获得有效补证，保留待查并继续其他工作。")
            else:
                gap_id = (active or feedback.get("arguments", {})).get("gap_id")
                gaps = self.a.data["gaps"]
                gap = gaps.get(gap_id) or next(
                    (
                        g
                        for g in gaps.values()
                        if g["status"] == "open" and g["kind"] in FOLLOWUP_GAP_KINDS
                    ),
                    None,
                )
                if gap:
                    gap["status"] = "stalled"
                    self.store.event(
                        "adaptive_gap_deferred",
                        gap_id=gap["gap_id"],
                        reason="此缺口暂未获得新证据，继续其他问题。",
                    )
            if active:
                self.a.close_round("Set aside the stalled item and continue with other gaps")
            if low_yield:
                low_yield["low_yield"] = 0
            self.data.update(progress=None, stagnant_turns=0)
            self.a.refresh()
        self.save()

    def context_signature(self):
        return digest(sorted((v.version_id, v.content) for v in self.evidence.documents.values()))

    def reject_repeat(self, call, reason):
        """Persist a blocked action through compaction; never pay for identical prompts."""
        if self.data["active"]:
            self.data["last_action_feedback"] = {
                "tool": call["name"],
                "arguments": call["args"],
                "error": reason,
            }
            if self.a.active():
                blocked = self.a.active().setdefault("blocked_lookup_tools", [])
                if call["name"] not in blocked:
                    blocked.append(call["name"])
                self.a.save()
            self.save()
        return reason

    def record_feedback(self, call, payload):
        if self.data["active"] and (
            payload.get("error")
            or payload.get("rejected_targets")
            or call["name"]
            in {
                "propose_temporal_resolution",
                "retain_temporal_unknown",
                "request_temporal_reaudit",
            }
        ):
            self.data["last_action_feedback"] = {
                "tool": call["name"],
                "arguments": call["args"],
                **{
                    k: payload[k]
                    for k in [
                        "error",
                        "status",
                        "accepted_target_ids",
                        "rejected_targets",
                        "deferred_target_ids",
                        "instruction",
                    ]
                    if k in payload
                },
            }
            self.save()

    def observe_context(self, payload):
        """Keep exact returned context visible after compaction, without certifying it."""
        passages = payload.get("passages", [])
        if payload.get("passage_id") and payload.get("text"):
            passages = [payload]
        inspected = set(self.data.get("inspected_context", []))
        recent = self.data.get("recent_context", [])
        for row in passages:
            pid = row.get("passage_id")
            source = self.evidence.passages.get(pid)
            if not source or row.get("text") != source.text:
                continue
            inspected.add(pid)
            recent = [r for r in recent if r["passage_id"] != pid]
            recent.append(
                {
                    "passage_id": pid,
                    "version_id": source.version_id,
                    "text": source.text[:1200],
                    "truncated": len(source.text) > 1200,
                }
            )
        self.data.update(inspected_context=sorted(inspected), recent_context=recent[-4:])
        self.save()

    def begin_context(self, gap_id):
        self.a.refresh()
        gap = self.a.data["gaps"].get(gap_id)
        if (
            not self.data["active"]
            or self.a.active()
            or not gap
            or gap["kind"] != "temporal_audit"
            or gap["status"] != "open"
        ):
            raise ValueError("select an open temporal audit gap with no active round")
        signature = self.context_signature()
        if any(
            r.get("local_context")
            and r["gap_id"] == gap_id
            and r.get("context_signature") == signature
            for r in self.a.data["rounds"]
        ):
            raise ValueError(
                "saved context round already attempted; search a different lead or retain unknown"
            )
        prior = self.a.previous_window(gap_id)
        r = {
            "round_id": "round-" + str(len(self.a.data["rounds"]) + 1),
            "gap_id": gap_id,
            "node_id": gap["node_id"],
            "strategy": "saved_context",
            "query": "",
            "window": prior["window"] if prior else {"precision": "unknown"},
            "movement": "retain" if prior else "unbounded",
            "status": "active",
            "local_context": True,
            "context_signature": signature,
            "before": self.a.measure(gap),
            "fingerprint": digest([gap_id, "saved_context", signature]),
            "search_result": {"candidate_ids": []},
            "actions": [],
        }
        self.a.data["rounds"].append(r)
        gap["attempts"] += 1
        self.a.save()
        self.store.event("temporal_context_started", round_id=r["round_id"], gap_id=gap_id)
        return self.summary()

    @staticmethod
    def originals_signature(row):
        return digest(sorted({(r["version_id"], r["quote"]) for r in row["references"]}))

    @staticmethod
    def prior_proposal(target, candidate, signature):
        return next(
            (
                p
                for p in target["proposals"]
                if p["finding_id"] == candidate["finding_id"]
                or p.get("originals_signature") == signature
            ),
            None,
        )

    def propose(self, target_ids, replacement_finding_id, reason):
        rows = self.findings()
        if not self.data["active"] or replacement_finding_id not in rows or not reason.strip():
            raise ValueError("select a saved source-backed replacement during temporal repair")
        node, row = rows[replacement_finding_id]
        if not row["references"] or not target_ids or len(target_ids) > 12:
            raise ValueError("replacement requires originals and 1-12 target IDs")
        accepted, rejected, deferred = [], [], []
        signature = self.originals_signature(row)
        for key in dict.fromkeys(target_ids):
            t = self.data["targets"].get(key)
            if (
                not t
                or t["status"] not in {"open", "awaiting_review"}
                or t["finding_id"] == replacement_finding_id
            ):
                rejected.append(
                    {
                        "target_id": key,
                        "error": "select an unresolved target and a different saved finding",
                    }
                )
            elif node["node_id"] != t["node_id"]:
                rejected.append(
                    {
                        "target_id": key,
                        "error": "different_investigation_node_requires_explicit_event_relation",
                        "required_node_id": t["node_id"],
                    }
                )
            elif prior := self.prior_proposal(t, row, signature):
                receipt = {"target_id": key, "error": "candidate_or_originals_already_proposed"}
                if t["status"] == "open" and (verdict := prior.get("rejected_by_review")):
                    self.defer_target(t, verdict["reason"])
                    receipt.update(
                        review_reason=verdict["reason"], target_status="retained_unknown"
                    )
                    deferred.append(key)
                rejected.append(receipt)
            else:
                accepted.append(key)
        for key in accepted:
            self.data["targets"][key]["proposals"].append(
                {
                    "finding_id": replacement_finding_id,
                    "reason": reason[:500],
                    "originals_signature": signature,
                    "scope": "temporal_only",
                    "candidate_question_part_id": row["finding"].get("question_part_id"),
                }
            )
            self.data["targets"][key]["status"] = "awaiting_review"
        self.save()
        if deferred:
            active = self.a.active()
            if (
                active
                and active.get("local_context")
                and any(
                    self.data["targets"][key]["node_id"] == active["node_id"] for key in deferred
                )
            ):
                self.a.close_round("Previously rejected evidence set aside for this field")
            self.a.refresh()
        self.store.event(
            "temporal_repair_proposed" if accepted else "temporal_repair_rejected",
            target_ids=accepted,
            rejected_targets=rejected,
            replacement_finding_id=replacement_finding_id,
        )
        return {
            "status": "awaiting_independent_reaudit" if accepted else "proposal_rejected",
            "resolved": False,
            "affects_answer_coverage": False,
            "accepted_target_ids": accepted,
            "rejected_targets": rejected,
            "deferred_target_ids": deferred,
            "instruction": (
                "Accepted targets are PENDING, not resolved. Call request_temporal_reaudit now; "
                "address rejected targets separately."
                if accepted
                else "Continue with the remaining open targets or question gaps. "
                "Use a different original for any open target. "
                "A retained_unknown field is set aside."
            ),
        }

    def retain_unknown(self, target_ids, reason):
        if not self.data["active"] or not reason.strip():
            raise ValueError("provide the observed limitation during active repair")
        for key in target_ids:
            t = self.data["targets"].get(key)
            if (
                not t
                or t["status"] not in {"open", "awaiting_review"}
                or not any(
                    r["node_id"] == t["node_id"]
                    and (
                        any(a["status"] in {"returned", "error"} for a in r["actions"])
                        or any(a.get("completed") for a in r.get("local_receipts", {}).values())
                        or not r.get("local_context")
                    )
                    for r in self.a.data["rounds"]
                    if self.a.data["gaps"][r["gap_id"]]["kind"] == "temporal_audit"
                )
            ):
                raise ValueError("inspect context or attempt retrieval before retaining unknown")
        for key in target_ids:
            self.data["targets"][key].update(status="retained_unknown", outcome_reason=reason[:500])
        self.save()
        return {"status": "retained_unknown", "resolved": False}

    def request_audit(self, reason):
        if not any(t["status"] == "awaiting_review" for t in self.data["targets"].values()):
            raise ValueError("no new replacement proposal to audit")
        if self.a.active():
            self.a.close_round(reason[:500])
        self.data["audit_requested"] = True
        self.save()
        self.budget.finish_research("temporal_reaudit_requested")
        return {
            "status": "reaudit_requested",
            "instruction": "Summarize pending evidence; auditor will run separately.",
        }

    def tools(self):
        def respond(fn, **args):
            try:
                return json.dumps(fn(**args), ensure_ascii=False)
            except (ValueError, KeyError) as exc:
                return json.dumps({"error": str(exc)}, ensure_ascii=False)

        @tool
        async def begin_temporal_context(gap_id: str) -> str:
            """Open a temporal audit gap for reading saved context, without a web search."""
            return respond(self.begin_context, gap_id=gap_id)

        @tool
        async def propose_temporal_resolution(
            target_ids: list[str], replacement_finding_id: str, reason: str
        ) -> str:
            """Link an existing OR newly saved finding to audit targets; never edit old dates.

            Use eligible_saved_candidate_ids to inspect same-node alternatives, even from a
            different question part. This changes only time links, never answer coverage. A
            separately dated record may clarify the same event while the original remains
            undated. Only the isolated re-audit can confirm identity/date and close the gap.
            """
            return respond(
                self.propose,
                target_ids=target_ids,
                replacement_finding_id=replacement_finding_id,
                reason=reason,
            )

        @tool
        async def retain_temporal_unknown(target_ids: list[str], reason: str) -> str:
            """Preserve unresolved time after attempted context/retrieval; no resolved label."""
            return respond(self.retain_unknown, target_ids=target_ids, reason=reason)

        @tool
        async def request_temporal_reaudit(reason: str) -> str:
            """Hand new replacement proposals to the isolated time auditor; no date editing."""
            return respond(self.request_audit, reason=reason)

        return [
            begin_temporal_context,
            propose_temporal_resolution,
            retain_temporal_unknown,
            request_temporal_reaudit,
        ]


async def complete_temporal_cycle(draft, task, context, model, budget, config, graph, router=None):
    """The same checkpointed Agent repairs gaps; the isolated auditor alone judges closure."""
    from . import event_report, temporal_review

    repair = context.evidence.temporal_repair
    visited = set()
    while True:
        plan = event_report.choices(context.evidence, context.report_packet())
        audit = await temporal_review.run_review(
            task, context.evidence, plan, model, budget, config, router
        )
        repair.ingest(audit)
        if audit.get("status") != "reviewed" or audit.get("binding") in visited:
            repair.stop("audit_unavailable_or_unchanged_evidence")
            break
        visited.add(audit["binding"])
        if not repair.start():
            break
        repair.store.save(
            f"temporal_drafts/{len(repair.data['reviews'])}.json",
            {"draft": draft, "audit_binding": audit["binding"]},
        )
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        "TEMPORAL_REPAIR_HANDOFF: A separate time audit found unresolved time "
                        "roles. Evidence work is reopened under the SAME remaining budget. Use "
                        "temporal_repair targets in ADAPTIVE_INVESTIGATION. Read saved context; "
                        "search missing evidence, append source-bound findings, explicitly link "
                        "their targets, and request re-audit. Preserve unknown if a year or "
                        "applicability cannot be established. Earlier final prose is an "
                        "unverified draft. "
                    )
                ]
            },
            config,
        )
        checkpoint = await graph.aget_state(config)
        if checkpoint.next:
            repair.stop("checkpoint_interrupted")
            break
        last = result["messages"][-1]
        if (
            last.type != "ai"
            or last.tool_calls
            or last.response_metadata.get("finish_reason") in {"length", "max_tokens"}
        ):
            repair.stop("repair_did_not_finalize")
            break
        draft = last.text
        repair.data["active"] = False
        repair.save()
    repair.data["active"] = False
    repair.save()
    return draft
