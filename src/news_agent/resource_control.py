"""Execution capacity, including explicit accounting-only diagnostic budgets."""

import sys


def execution_limits(config):
    extended = (
        config.max_tokens is None
        or config.max_tokens > 80000
        or config.max_model_calls is None
        or config.max_model_calls > 24
    )
    return {
        "version": "extended-budget-v1",
        # LangGraph requires a positive integer. This engine sentinel imposes no
        # practical turn budget; it is not used for accounting/admission.
        "recursion_limit": sys.maxsize
        if config.max_model_calls is None
        else max(80, 4 * config.max_model_calls + 16)
        if extended
        else 80,
        "stagnation_warning_turns": 6 if extended else None,
        "stagnation_action": "log_only_no_forced_stop",
        "model_timeout_seconds": 300 if extended else 60,
        "low_yield_rounds_per_gap": 6 if extended else 2,
        "evidence_packet_bytes": 24000
        if config.max_tokens is None
        else min(24000, config.max_tokens // 20)
        if config.max_tokens > 80000
        else 6000,
        "max_output_tokens": config.max_output_tokens,
        "progress": "new delivered passages, graph findings, source relations, candidate versions",
    }


def check_progress(adaptive, budget):
    """A model turn is not itself progress. State persists across resume."""
    cap = execution_limits(budget.config)["stagnation_warning_turns"]
    if cap is None or budget.state["phase"] != "research":
        return
    graph = adaptive.graph.data
    signature = {
        "passages": sorted(budget.state.get("delivered_passages", [])),
        "findings": sorted(f["finding_id"] for n in graph["nodes"].values() for f in n["findings"]),
        "source_relations": len(graph["source_relations"]),
        "candidate_versions": sorted(budget.state.get("seen_search_versions", [])),
    }
    previous = budget.state.get("extended_progress", {})
    stalled = previous.get("stalled", 0) + 1 if previous.get("signature") == signature else 0
    budget.state["extended_progress"] = {"signature": signature, "stalled": stalled}
    budget.save()
    if stalled and stalled % cap == 0:
        budget.store.event("extended_no_progress_warning", stagnant_turns=stalled, interval=cap)
