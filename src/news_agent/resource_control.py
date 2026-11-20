"""Execution capacity, including explicit accounting-only diagnostic budgets."""

import sys

from .event_graph import digest


def execution_limits(config):
    extended = (
        config.max_tokens is None
        or config.max_tokens > 80000
        or config.max_model_calls is None
        or config.max_model_calls > 24
    )
    return {
        "version": "extended-budget-v2",
        # LangGraph requires a positive integer. This engine sentinel imposes no
        # practical turn budget; it is not used for accounting/admission.
        "recursion_limit": sys.maxsize
        if config.max_model_calls is None
        else max(80, 4 * config.max_model_calls + 16)
        if extended
        else 80,
        "stagnation_warning_turns": 6 if extended else None,
        "stagnation_action": "defer_active_gap_then_write",
        "model_timeout_seconds": 300 if extended else 60,
        "low_yield_rounds_per_gap": 6 if extended else 2,
        "evidence_packet_bytes": 24000
        if config.max_tokens is None
        else min(24000, config.max_tokens // 20)
        if config.max_tokens > 80000
        else 6000,
        "max_output_tokens": config.max_output_tokens,
        "progress": (
            "new passages, evidence bindings, part coverage, source relations, candidate versions"
        ),
    }


def check_progress(adaptive, budget):
    """A model turn is not itself progress. State persists across resume."""
    cap = execution_limits(budget.config)["stagnation_warning_turns"]
    if cap is None or budget.state["phase"] != "research" or adaptive.current_check:
        return
    if adaptive.temporal_repair.data["active"]:
        # Temporal repair has its own field-level scheduler and stagnation policy.
        budget.state.pop("extended_progress", None)
        return
    graph = adaptive.graph.data
    views = {nid: adaptive.graph.node_view(nid) for nid in graph["nodes"]}
    snapshot = {
        "passages": sorted(budget.state.get("delivered_passages", [])),
        "bindings": [
            [
                nid,
                sorted(
                    {
                        (
                            ref["passage_id"],
                            ref["start"],
                            ref["end"],
                            f["finding"]["event_time"]["start"] or "",
                            f["finding"]["event_time"]["end"] or "",
                            f["finding"].get("monthly_value") or 0,
                        )
                        for f in node["findings"]
                        for ref in f["references"]
                    }
                ),
                views[nid]["status"],
                views[nid]["question_coverage"],
                [(p["part_id"], p["status"]) for p in views[nid].get("part_status", [])],
            ]
            for nid, node in sorted(graph["nodes"].items())
        ],
        "source_relations": len(graph["source_relations"]),
        "candidate_versions": sorted(budget.state.get("seen_search_versions", [])),
        "scoped_searches": sorted(
            (g["gap_id"], g["attempts"], g["status"])
            for g in adaptive.data["gaps"].values()
            if g.get("question_part_id")
        ),
        "replans": [
            (r["round_id"], r["replan"]["status"])
            for r in adaptive.data["rounds"]
            if r.get("replan")
        ],
    }
    signature = digest(snapshot)
    previous = budget.state.get("extended_progress", {})
    stalled = previous.get("stalled", 0) + 1 if previous.get("signature") == signature else 0
    budget.state["extended_progress"] = {"signature": signature, "stalled": stalled}
    budget.save()
    if stalled == cap:
        budget.store.event("extended_no_progress_warning", stagnant_turns=stalled, interval=cap)
    if stalled < 2 * cap:
        return
    if current := adaptive.active():
        reason = "当前缺口连续操作未获得可用新证据，保留待查并继续其他问题。"
        adaptive.close_round(reason)
        gap = adaptive.data["gaps"][current["gap_id"]]
        if gap["status"] == "open" and current.get("replan", {}).get("status") != "pending":
            gap["status"] = "stalled"
            adaptive.save()
            budget.store.event("adaptive_gap_deferred", gap_id=gap["gap_id"], reason=reason)
        budget.state["extended_progress"]["stalled"] = 0
        budget.save()
    else:
        budget.finish_research("adaptive_no_progress")
