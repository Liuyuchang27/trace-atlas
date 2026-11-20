"""Bounded plan corrections from recorded retrieval outcomes."""

import re

from .event_graph import digest


def queue_replan(adaptive, round_, gap):
    if (
        adaptive.budget.state["phase"] != "research"
        or adaptive.temporal_repair.data["active"]
        or gap["status"] != "open"
        or any(r.get("replan") and r["gap_id"] == gap["gap_id"] for r in adaptive.data["rounds"])
    ):
        return
    search = round_["search_result"]
    candidates = search.get("candidate_ids", [])
    passages = sorted(adaptive.round_originals(round_))
    failure = (
        "search_error"
        if search.get("error")
        else "no_candidates"
        if not candidates
        else "no_answer_bound"
        if passages
        else "original_not_obtained"
    )
    receipt = {
        "round_id": round_["round_id"],
        "gap_id": gap["gap_id"],
        "question": gap["title"],
        "status": "pending",
        "failure": failure,
        "search_error": search.get("error"),
        "candidate_ids": candidates[:6],
        "original_passage_ids": passages[:8],
        "errors": [a["error"] for a in round_["actions"] if a.get("error")][-3:],
        "failed_query": round_["query"],
        "previous_strategy": round_["strategy"],
        "previous_window": round_["window"],
        "source_domains": round_.get("include_domains") or [],
        "can_split": not gap.get("parent_gap_id"),
    }
    if gap.get("reference_month"):
        receipt["reference_month"] = gap["reference_month"]
    round_["replan"] = receipt
    adaptive.store.event("adaptive_replan_triggered", **receipt)
    adaptive.save()


def pending_replan(adaptive):
    if (
        adaptive.budget.state["phase"] != "research"
        or adaptive.active()
        or adaptive.temporal_repair.data["active"]
    ):
        return None
    return next(
        (
            receipt
            for round_ in adaptive.data["rounds"]
            if (receipt := round_.get("replan"))
            and receipt["status"] == "pending"
            and adaptive.data["gaps"][receipt["gap_id"]]["status"] == "open"
        ),
        None,
    )


def apply_replan(adaptive, gap_id, route, reason, questions, source_domains=None):
    receipt = pending_replan(adaptive)
    if not receipt or receipt["gap_id"] != gap_id:
        raise ValueError("Choose the gap with a pending search outcome")
    if not reason.strip() or len(reason) > 320:
        raise ValueError("Explain the next route in relation to the failed outcome")
    parent = adaptive.data["gaps"][gap_id]
    children = []
    if route == "split":
        keys = [" ".join(q.casefold().split()) for q in questions]
        if (
            not receipt["can_split"]
            or not 2 <= len(questions) <= 3
            or any(not k or len(q) > 160 for k, q in zip(keys, questions, strict=True))
            or len(set(keys)) != len(keys)
            or " ".join(parent["title"].casefold().split()) in keys
        ):
            raise ValueError("Split once into two or three distinct, narrower evidence questions")
        for question, key in zip(questions, keys, strict=True):
            child_id = "gap-" + digest([gap_id, key])[:14]
            adaptive.data["gaps"][child_id] = {
                **{
                    k: parent[k]
                    for k in (
                        "node_id",
                        "kind",
                        "importance",
                        "question_part_id",
                        "reference_month",
                    )
                    if k in parent
                },
                "gap_id": child_id,
                "parent_gap_id": gap_id,
                "title": question.strip(),
                "status": "open",
                "attempts": 0,
                "low_yield": 0,
                "evidence_signature": None,
            }
            children.append(child_id)
        parent["replan_children"] = children
        parent["status"] = "stalled"
    elif route == "continue_other":
        parent["status"] = "stalled"
    elif route == "change_route":
        domains = sorted({domain.strip().casefold() for domain in source_domains or []})
        if (
            not 1 <= len(domains) <= 8
            or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", d) for d in domains)
            or set(domains) == {d.casefold().strip() for d in receipt["source_domains"]}
        ):
            raise ValueError("Choose different source_domains to execute this route change")
        receipt["next_source_domains"] = domains
    receipt.update(status="decided", route=route, reason=reason, children=children)
    adaptive.refresh()
    adaptive.store.event("adaptive_replan_decided", **receipt)
    return receipt


def route_change(adaptive, gap_id):
    for round_ in adaptive.data["rounds"]:
        receipt = round_.get("replan", {})
        if (
            receipt.get("gap_id") == gap_id
            and receipt.get("route") == "change_route"
            and receipt["status"] == "decided"
            and receipt.get("next_source_domains")
        ):
            return receipt
    return None
