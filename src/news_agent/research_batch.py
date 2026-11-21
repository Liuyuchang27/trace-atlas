"""Bounded parallel searches for independent evidence targets, followed by joint binding."""

import asyncio
import json
import re
from calendar import month_name

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import Field

from .event_graph import TimeSpan, digest
from .monthly_series import explicit_search_months
from .schemas import StrictModel

MAX_CONCURRENT_SEARCHES = 42
NOT_DISPATCHED = {"research_finished", "deadline_reached", "tool_call_limit", "round_limit"}

SCOPE_PLAN = f"""You plan investigations for a deep research agent. Start with what the user
wants to learn. Work backwards from each conclusion to the evidence that could establish it.
Assign the web the job of locating observable inputs. Assign the agent the work of
extracting, comparing like with like, computing changes, resolving discrepancies, and
writing a sourced answer. Save those analytical operations in scope.synthesis_tasks.

Choose retrievable units from the way the evidence is actually published: an edition of
a release, a filing for an entity and period, a decision, a dated announcement, or a
comparable row in an archive. Name the original record in the publisher's language and
state who publishes it, how it can be accessed, and an alternative archive or reporting
route. Treat expected availability as a hypothesis until the original has been read.
You have web search with domain filters, URL and original-page reading, saved-text lookup,
source-link traversal, evidence binding and a report writer. Search snippets are leads.
Prefer publicly readable records, downloadable tables and issuer archives. If a page is
inaccessible or superseded, try another path and record which coverage remains missing.
Do not outsource a synthesis question to a search engine. Do not invent URLs or dates.

Infer the time horizon, publication cadence, measure, units, geography and comparison
basis from the question. For monthly observations use monthly_measures with periods,
publication and unit. The runtime finds the latest published observation month, expands
the period slots, shares one edition across metrics, and can search/read up to
{MAX_CONCURRENT_SEARCHES} originals concurrently. For individual records or event histories
use records with distinct evidence_needed and genuine dependencies. A source-independent
explanation can wait for its underlying records. A single original table can establish many periods and measures. Read it once, bind each supported cell and dispatch only remaining gaps. Use month-independent metric labels, without provisional dates. Plan exactly the requested number of OUTPUT observations; any earlier comparison base is an input only. For changes in a published index choose the publisher's headline period-on-period and year-on-year changes when available, with adjustment and geography stated. Prefer the publisher's measured changes over computing extra cumulative or annualized indicators. Add a level series only when the user asks for the level. Monthly plans already create an edition-verification task: never duplicate the latest month/date or lookup routes in records. Put table selection and finding the latest release in access_route. Domain means bare host. Do not predetermine the range before verifying the actual release. Verify the latest observed period using a current edition and the publisher calendar. An overdue forecast triggers a search for a later release.
Never expand a simple request into unrequested topics. Keep the plan in the user's language
except for search titles. Return a concise public plan with no private reasoning."""


class EvidenceSearch(StrictModel):
    gap_id: str
    query: str = Field(min_length=1, max_length=520)
    include_domains: list[str] | None = Field(default=None, max_length=8)
    limit: int = Field(default=5, ge=1, le=10)


class MonthlyRecord(StrictModel):
    record_query: str = Field(min_length=1, max_length=520)
    question_part_ids: list[str] = Field(min_length=1)
    include_domains: list[str] | None = Field(default=None, max_length=8)


class EvidenceRead(StrictModel):
    gap_id: str
    version_id: str


async def run_parallel(invoke, arguments):
    slots = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def limited(args):
        async with slots:
            return await invoke(*args)

    async with asyncio.TaskGroup() as workers:
        jobs = [workers.create_task(limited(args)) for args in arguments]
    return [job.result() for job in jobs]


def month_query(query, reference_month):
    year, month = reference_month.split("-")
    return f"{query.strip()} [reference month: {month_name[int(month)]} {year}; {reference_month}]"


def search_fingerprint(query, domains):
    return digest(
        [" ".join(query.casefold().split()), sorted(d.strip().casefold() for d in domains or [])]
    )


def attempted_search(rounds, fingerprint):
    return any(
        row.get("fingerprint") == fingerprint
        or any(
            target["fingerprint"] == fingerprint
            and target["status"] not in {"planned", "not_dispatched"}
            for target in row.get("batch_targets", [])
        )
        for row in rounds
    )


def batch_outcomes(adaptive):
    return {
        target["gap_id"]: target
        for row in adaptive.data["rounds"]
        if row["status"] == "closed"
        for target in row.get("batch_targets", [])
    }


def pending_targets(adaptive):
    if (
        not adaptive.parallel_search_enabled
        or adaptive.data.get("batch_fallback_round_count") == len(adaptive.data["rounds"])
        or adaptive.active()
        or adaptive.budget.state["phase"] != "research"
        or adaptive.temporal_repair.data["active"]
    ):
        return []
    outcomes = batch_outcomes(adaptive)
    groups = {}
    for gap in adaptive.data["gaps"].values():
        if (
            gap["status"] == "open"
            and not gap.get("blocked_by")
            and (not gap["attempts"] or outcomes.get(gap["gap_id"], {}).get("covered") is False)
            and not gap.get("parent_gap_id")
            and gap.get("question_part_id")
            and gap["kind"] == "question_or_explanation_unresolved"
        ):
            groups.setdefault(gap["node_id"], []).append(gap)
    for gaps in groups.values():
        ready = [g for g in gaps if g.get("reference_month") or g.get("evidence_needed")]
        if len(ready) > 1:
            return sorted(
                ready,
                key=lambda g: (g["attempts"], g.get("reference_month", ""), g["question_part_id"]),
            )
    return []


def target_receipt(target):
    return {
        key: target[key]
        for key in (
            "gap_id",
            "question_part_id",
            "reference_month",
            "status",
            "candidate_ids",
            "error",
            "covered",
            "read_versions",
            "passage_ids",
        )
        if key in target
    }


def recover_batch(row):
    for target in row["batch_targets"]:
        if target["status"] == "dispatching":
            target.update(status="interrupted_unknown", error="search_result_not_committed")
        elif target["status"] == "planned":
            target["status"] = "not_dispatched"
    candidate_ids = list(
        dict.fromkeys(
            vid for target in row["batch_targets"] for vid in target.get("candidate_ids", [])
        )
    )
    row["search_result"] = {"candidate_ids": candidate_ids, "error": None}
    row["status"] = "active" if candidate_ids else "interrupted_unknown"


def batch_can_read(row):
    if not row or row.get("batch_read_handoff") or not row.get("batch_targets"):
        return False
    seen = {read["version_id"] for read in row.get("batch_reads", [])}
    return any(set(target.get("candidate_ids", [])) - seen for target in row["batch_targets"])


def close_batch(adaptive, row, reason):
    view = adaptive.graph.node_view(row["node_id"])
    parts = {p["part_id"]: p for p in view["part_status"]}
    covered_ids = set()
    for target in row["batch_targets"]:
        gap = adaptive.data["gaps"][target["gap_id"]]
        part = parts[target["question_part_id"]]
        if month := target.get("reference_month"):
            point = part["month_coverage"]["points"].get(month)
            target["covered"] = point is not None
            if point:
                covered_ids.add(point["finding_id"])
        else:
            target["covered"] = part["status"] == "model_assessed_answered"
            if target["covered"]:
                covered_ids.update(part["finding_ids"])
        if target["covered"]:
            gap["low_yield"] = 0
        elif target["status"] != "not_dispatched":
            gap["low_yield"] += 1
    originals = adaptive.round_originals(row)
    bound = any(
        finding["finding_id"] in covered_ids
        and any(ref["passage_id"] in originals for ref in finding["references"])
        for finding in view["findings"]
    )
    gain = any(t["covered"] for t in row["batch_targets"])
    row.update(
        status="closed",
        close_reason=reason,
        evidence_gain=gain,
        current_round_originals_bound=bound,
        retrieval_evidence_gain=bound,
    )
    adaptive.refresh()
    result = {
        "round_id": row["round_id"],
        "evidence_gain": gain,
        "targets": [target_receipt(t) for t in row["batch_targets"]],
    }
    adaptive.store.event("adaptive_batch_closed", **result, reason=reason)
    return result


def batch_tool(adaptive):
    @tool
    async def locate_latest_edition(config: RunnableConfig) -> str:
        """Search for the planned publisher's latest original release before expanding months."""
        pending = adaptive.initial_publication()
        if not pending:
            return json.dumps({"error": "Latest edition lookup is not pending"})
        gap, spec = pending
        result = await adaptive.search_gap(
            gap["gap_id"],
            spec["publication"] + " latest release",
            "primary_record",
            TimeSpan(),
            "unbounded",
            "核对最新一期原文的数据所属月",
            config,
            include_domains=[spec["domain"]] if spec.get("domain") else None,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool
    async def search_planned_months(config: RunnableConfig) -> str:
        """Dispatch one original-publication search per outstanding observation month now."""
        gaps = [g for g in pending_targets(adaptive) if g.get("publication") and not g["attempts"]]
        if not gaps:
            return json.dumps({"error": "No untried monthly editions"})
        entries = {
            g["gap_id"]: EvidenceSearch(
                gap_id=g["gap_id"],
                query=g["publication"],
                include_domains=[g["domain"]] if g.get("domain") else None,
            )
            for g in gaps
        }
        return await execute(gaps, entries, "按数据所属月查找原始发布", config)

    @tool
    async def search_monthly_records(
        record_query: str,
        question_part_ids: list[str],
        config: RunnableConfig,
        include_domains: list[str] | None = None,
        additional_records: list[MonthlyRecord] | None = None,
    ) -> str:
        """Find the original monthly publication or dataset for the named question parts.

        Supply one source-language publication name and geography, e.g. United States
        Consumer Price Index release. Choose parts sharing this publication. The scheduler
        inserts every outstanding month and shares each result across those parts. Extract
        the requested measures, adjustments and trend from the originals after reading.
        """
        adaptive.refresh()
        offered = [g for g in pending_targets(adaptive) if g.get("reference_month")]
        from pydantic import ValidationError

        try:
            first = MonthlyRecord(
                record_query=record_query.strip(),
                question_part_ids=question_part_ids,
                include_domains=include_domains,
            )
        except ValidationError:
            return json.dumps(
                {"error": "Provide a publication name, part IDs and up to eight hostnames"}
            )
        entries = {}
        known = {g["question_part_id"] for g in offered}
        for record in [first, *(additional_records or [])]:
            ids = set(record.question_part_ids)
            if ids - known or explicit_search_months(record.record_query):
                return json.dumps(
                    {
                        "error": "Use offered part IDs and an undated publication name",
                        "available": sorted(known),
                    }
                )
            for gap in offered:
                gid = gap["gap_id"]
                if gap["question_part_id"] not in ids:
                    continue
                if gid in entries:
                    return json.dumps({"error": "Assign each part to one publication per round"})
                entries[gid] = EvidenceSearch(
                    gap_id=gid, query=record.record_query, include_domains=record.include_domains
                )
        return await execute(
            [g for g in offered if g["gap_id"] in entries], entries, first.record_query, config
        )

    @tool
    async def search_evidence_batch(
        searches: list[EvidenceSearch], reason: str, config: RunnableConfig
    ) -> str:
        """Search independent original records for listed gap IDs; a subset is allowed."""
        adaptive.refresh()
        entries = {s.gap_id: s for s in searches}
        gaps = [g for g in pending_targets(adaptive) if g["gap_id"] in entries]
        if not gaps or len(entries) != len(searches) or len(gaps) != len(entries):
            return json.dumps(
                {
                    "error": "Choose distinct offered gap IDs",
                    "available": [g["gap_id"] for g in pending_targets(adaptive)],
                }
            )
        return await execute(gaps, entries, reason.strip()[:240], config)

    async def execute(gaps, entries, reason, config):
        targets = []
        groups = {}
        for gap in gaps:
            search = entries[gap["gap_id"]]
            month = gap.get("reference_month")
            query = search.query.strip()
            if not query or (month and explicit_search_months(query) - {month}):
                return json.dumps(
                    {"error": "Give a nonempty query; use only the selected observation month"}
                )
            domains = sorted({d.strip().casefold() for d in search.include_domains or []})
            if any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", d) for d in domains):
                return json.dumps({"error": "Source filters must be hostnames"})
            if month and not explicit_search_months(query):
                query = month_query(query, month)
            fingerprint = search_fingerprint(query, domains)
            if attempted_search(adaptive.data["rounds"], fingerprint):
                continue
            targets.append(
                {
                    "gap_id": gap["gap_id"],
                    "question_part_id": gap["question_part_id"],
                    **({"reference_month": month} if month else {}),
                    "question": gap["title"],
                    "evidence_needed": gap.get("evidence_needed"),
                    "query": query,
                    "record_name": search.query,
                    "include_domains": domains,
                    "limit": search.limit,
                    "fingerprint": fingerprint,
                    "status": "planned",
                }
            )
            group = groups.setdefault(fingerprint, [])
            targets[-1]["search_group"] = group[0][0]["gap_id"] if group else gap["gap_id"]
            group.append((gap, targets[-1]))
        if not targets:
            return json.dumps({"error": "Queries already ran; change that target's evidence route"})

        first = next(iter(groups.values()))[0][0]
        parts = {t["question_part_id"] for t in targets}
        budget, store = adaptive.budget, adaptive.store
        round_id = f"round-{len(adaptive.data['rounds']) + 1}"
        row = {
            "round_id": round_id,
            "gap_id": first["gap_id"],
            "node_id": first["node_id"],
            "question_part_id": next(iter(parts)) if len(parts) == 1 else None,
            "query": reason,
            "strategy": "primary_record",
            "window": TimeSpan().model_dump(),
            "movement": "parallel_targets",
            "reason": reason,
            "status": "dispatching",
            "action_id": budget.state.get("active_tool_action_id"),
            "batch_targets": targets,
            "actions": [],
            "search_result": {},
        }
        adaptive.data["rounds"].append(row)
        adaptive.save()
        store.event(
            "adaptive_batch_started",
            round_id=round_id,
            action_id=row["action_id"],
            reason=reason,
            concurrency=MAX_CONCURRENT_SEARCHES,
            targets=[{k: v for k, v in t.items() if k != "fingerprint"} for t in targets],
        )

        async def fetch(group):
            for gap, target in group:
                target["status"] = "dispatching"
                gap["attempts"] += 1
            _, target = group[0]
            adaptive.save()
            arguments = {
                "query": target["query"],
                "purpose": target["question"][:240],
                "intent": "temporal" if target.get("reference_month") else "gap",
                "limit": target["limit"],
            }
            if target["include_domains"]:
                arguments["include_domains"] = target["include_domains"]
            result = json.loads(await adaptive.search.ainvoke(arguments, config=config))
            hits, error = result.get("results", []), result.get("error")
            for gap, member in group:
                if error in NOT_DISPATCHED:
                    gap["attempts"] -= 1
                member.update(
                    status="not_dispatched"
                    if error in NOT_DISPATCHED
                    else "error"
                    if error
                    else "returned",
                    error=error,
                    candidate_ids=[h["version_id"] for h in hits],
                )
            adaptive.save()
            for _, member in group:
                store.event(
                    "adaptive_batch_target_result", round_id=round_id, **target_receipt(member)
                )
            return hits

        results = await run_parallel(fetch, [(group,) for group in groups.values()])
        candidates = {hit["version_id"]: hit for hits in results for hit in hits}
        row["search_result"] = {"candidate_ids": list(candidates), "error": None}
        row["status"] = "active"
        if budget.state.get("evidence_chain_version"):
            budget.state["last_search_results"] = list(candidates.values())
            budget.save()
        adaptive.save()
        return json.dumps(
            {
                "round_id": round_id,
                "targets": [target_receipt(t) for t in targets],
                "candidates": [
                    {k: h[k] for k in ("version_id", "title", "url") if k in h}
                    for h in candidates.values()
                ],
                "instruction": (
                    "Read relevant originals once and bind each supported target separately. "
                    "Close the batch when ready, then change route for remaining gaps."
                ),
            },
            ensure_ascii=False,
        )

    @tool
    async def read_evidence_batch(reads: list[EvidenceRead], config: RunnableConfig) -> str:
        """Read selected originals for the active investigative lines concurrently.

        Choose a returned version_id for each gap_id within the advertised concurrency limit.
        A page selected by multiple lines is retrieved once. Bind each supported answer to
        the correct question part/observation month after the receipts return.
        """
        row = adaptive.active()
        if (
            not adaptive.reader
            or not batch_can_read(row)
            or not 1 <= len(reads) <= MAX_CONCURRENT_SEARCHES
        ):
            return json.dumps({"error": "Select unread candidates for the active batch"})
        targets = {t["gap_id"]: t for t in row["batch_targets"]}
        seen = {r["version_id"] for r in row.get("batch_reads", [])}
        version_targets = {}
        for read in reads:
            target = targets.get(read.gap_id, {})
            if read.version_id in seen or read.version_id not in target.get("candidate_ids", []):
                continue
            shared = target.get("search_group")
            linked = {
                gid
                for gid, other in targets.items()
                if shared and other.get("search_group") == shared
            }
            version_targets.setdefault(read.version_id, set()).update(linked | {read.gap_id})
        if not version_targets:
            row["batch_read_handoff"] = True
            adaptive.save()
            return json.dumps(
                {
                    "error": "No unread selection. Bind saved originals now or close this round.",
                    "saved_passages": sorted(adaptive.round_originals(row)),
                }
            )
        budget, store = adaptive.budget, adaptive.store
        action_id = budget.state.get("active_tool_action_id")
        plans = [
            {
                "version_id": vid,
                "gap_ids": sorted(ids),
                **{
                    k: budget.state["search_candidates"][vid][k]
                    for k in ("title", "url")
                    if k in budget.state.get("search_candidates", {}).get(vid, {})
                },
            }
            for vid, ids in version_targets.items()
        ]
        store.event(
            "adaptive_batch_read_started",
            round_id=row["round_id"],
            action_id=action_id,
            concurrency=MAX_CONCURRENT_SEARCHES,
            targets=plans,
        )

        async def read_original(plan):
            receipt = {**plan, "status": "dispatching", "action_id": action_id}
            row.setdefault("batch_reads", []).append(receipt)
            adaptive.save()
            result = json.loads(
                await adaptive.reader.ainvoke(
                    {"version_id": plan["version_id"], "passage_limit": 4}, config=config
                )
            )
            passages = result.get("passages", [])
            pids = result.get("passage_ids") or [p["passage_id"] for p in passages]
            error = result.get("error")
            receipt.update(status="error" if error else "returned", error=error, passage_ids=pids)
            for gid in plan["gap_ids"]:
                target = targets[gid]
                target.setdefault("read_versions", []).append(plan["version_id"])
                target["passage_ids"] = list(dict.fromkeys(target.get("passage_ids", []) + pids))
            adaptive.save()
            store.event("adaptive_batch_read_result", round_id=row["round_id"], **receipt)
            return {**receipt, "passages": [p for p in passages[:2] if len(p["text"]) <= 900]}

        originals = await run_parallel(read_original, ((plan,) for plan in plans))
        delivered = {pid for result in originals for pid in result["passage_ids"]}
        row["actions"].append(
            {
                "name": "read_evidence_batch",
                "status": "returned",
                "action_id": action_id,
                "passage_ids": sorted(delivered),
                "fingerprint": digest(["read_evidence_batch", plans]),
            }
        )
        adaptive.save()
        return json.dumps(
            {
                "originals": originals,
                "passages": [{"passage_id": pid} for pid in sorted(delivered)],
                "instruction": "Bind each supported answer to saved originals and close the batch.",
            },
            ensure_ascii=False,
        )

    return [
        locate_latest_edition,
        search_planned_months,
        search_monthly_records,
        search_evidence_batch,
        read_evidence_batch,
    ]
