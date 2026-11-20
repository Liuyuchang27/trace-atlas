"""Agent tool for finding, reading and binding a single monthly source version."""

import json
import re
from typing import Literal
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from langchain_core.tools import tool

from .budget import BudgetExceeded
from .event_graph import Finding, OriginalRef, TimeSpan, storage_limits
from .evidence_chain import expose_passages
from .historical_series import (
    FRED_HOST,
    RepositorySeriesLinks,
    extract_monthly_changes,
    fred_path,
    history_link,
    normalize_title_terms,
)
from .retrieval import RetrievalFailure


async def search_series_catalog(query, corpus, budget, store):
    """Retrieve the catalog's own ranked series links without supplying an identifier."""
    url = f"https://{FRED_HOST}/searchresults?{urlencode({'st': query})}"
    request_id = uuid4().hex
    args = {"query": query, "url": url}
    for key in ("web_search_requests", "retrieval_api_calls"):
        budget.state[key] = budget.state.get(key, 0) + 1
    budget.save()
    log = {"request_id": request_id, "operation": "search", "provider": "fred_public_catalog"}
    store.event("retrieval_dispatch", **log, arguments=args)
    status, error = None, None
    try:
        async with httpx.AsyncClient(timeout=min(15, budget.remaining_time())) as client:
            async with client.stream("GET", url) as response:
                status = response.status_code
                response.raise_for_status()
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > 1024**2:
                        raise RetrievalFailure("fred_catalog_response_too_large")
        parser = RepositorySeriesLinks()
        parser.feed(data.decode("utf-8", errors="replace"))
        hits = []
        for link, title in parser.links.items():
            vid = corpus.register(link, title)
            hits.append({"version_id": vid, "url": link, "title": title})
        corpus.persist()
        store.save(
            f"web_responses/{request_id}.json",
            {**log, "arguments": args, "response": {"results": hits}},
        )
        return hits
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        store.event(
            "retrieval_result", **log, status_code=status, error=error, credits=0, usage_known=True
        )


def monthly_history_tools(adaptive, corpus):
    graph, budget, evidence = adaptive.graph, adaptive.budget, adaptive.evidence

    def event(kind, **fields):
        adaptive.store.event(kind, action_id=budget.state.get("active_tool_action_id"), **fields)

    def unresolved(attempt, exc):
        reason = str(exc) if isinstance(exc, (ValueError, BudgetExceeded)) else type(exc).__name__
        attempt.update(status="unresolved", reason=reason)
        adaptive.refresh()
        event("monthly_series_unresolved", reason=reason)
        return json.dumps({"status": "unresolved", "reason": reason}, ensure_ascii=False)

    async def discover(query, publisher_domain, catalog_query):
        failures = []
        for route, query_text, domain in (
            ("publisher", query, publisher_domain),
            ("repository_catalog", catalog_query, FRED_HOST),
            ("repository_web_search", f"{query} FRED monthly data", FRED_HOST),
        ):
            if route != "publisher":
                event(
                    "monthly_series_route_changed"
                    if route == "repository_catalog"
                    else "monthly_series_lookup_refined",
                    reason="；".join(failures),
                    domain=domain,
                    query=query_text,
                )
            budget.before_tool("search")
            try:
                if route == "repository_catalog":
                    hits = await search_series_catalog(query_text, corpus, budget, adaptive.store)
                else:
                    hits = await corpus.asearch(
                        query_text,
                        limit=8,
                        timeout=min(30, budget.remaining_time()),
                        include_domains=[domain] if domain else None,
                    )
            except (RetrievalFailure, httpx.HTTPError) as exc:
                failures.append(
                    str(exc) if isinstance(exc, RetrievalFailure) else type(exc).__name__
                )
                continue
            # The search rank and displayed titles are leads. The Agent decides which
            # measure it asked for; metadata validation happens after original reading.
            candidates = [
                {k: hit[k] for k in ("version_id", "url", "title")}
                for hit in hits
                if fred_path(hit["url"])
                and not re.match(r"^(?:daily|weekly|monthly|quarterly|annual),", hit["title"], re.I)
            ][:12]
            event(
                "monthly_series_candidates",
                route=route,
                domain=domain,
                query=query_text,
                sources=[{"title": hit["title"], "url": hit["url"]} for hit in hits[:12]],
                candidate_count=len(candidates),
            )
            if candidates:
                return candidates
            failures.append("当前来源未返回可读取的历史序列入口")
        raise ValueError("；".join(failures))

    @tool
    async def collect_monthly_history(
        query: str,
        title_terms: list[str],
        publisher_terms: list[str],
        method: Literal["monthly_net_change", "per_month"],
        publisher_domain: str | None = None,
        adjustment: Literal["seasonally_adjusted", "not_seasonally_adjusted", "any"] = "any",
    ) -> str:
        """Discover official candidates for one full monthly historical table.

        Use monthly_net_change for absolute month-to-month differences. Give a source-language
        web query for the metric's historical levels and put its short, distinctive noun phrase
        FIRST in title_terms for catalog search. Specify the original publisher's name in
        publisher_terms and its domain if known. Set adjustment separately. Candidate titles
        come from actual search results and will be available to select_monthly_series.
        Use per_month for other measures that need ordinary searches.
        """
        target = adaptive.pending_series()
        if not target:
            return json.dumps({"error": "没有待处理的月度序列"}, ensure_ascii=False)
        node_id, part = target
        limit = storage_limits(budget.config)["max_findings_per_node"]
        attempt = {"node_id": node_id, "question_part_id": part["part_id"], "status": "started"}
        adaptive.data.setdefault("series_attempts", []).append(attempt)
        adaptive.save()
        if method == "per_month":
            attempt.update(status="per_month")
            adaptive.save()
            return json.dumps({"status": "per_month"}, ensure_ascii=False)
        try:
            if not (
                2 <= len(query.strip()) <= 320
                and all(1 <= len(terms) <= 6 for terms in (title_terms, publisher_terms))
                and all(1 <= len(term.strip()) <= 80 for term in [*title_terms, *publisher_terms])
            ):
                raise ValueError("历史序列检索参数超出范围")
            terms, adjustment = normalize_title_terms(title_terms, adjustment)
            spec = part["monthly_series"]
            if (
                limit is not None
                and len(graph.data["nodes"][node_id]["findings"]) + spec["periods"] > limit
            ):
                raise ValueError("月度序列超出当前图谱节点容量")
            candidates = await discover(query, publisher_domain, terms[0])
            attempt.update(
                status="choose_source",
                candidates=candidates,
                publisher_terms=publisher_terms,
                adjustment=adjustment,
                spec=spec,
            )
            adaptive.save()
            return json.dumps(
                {"status": "choose_source", "candidates": candidates},
                ensure_ascii=False,
            )
        except (BudgetExceeded, ValueError, httpx.HTTPError) as exc:
            return unresolved(attempt, exc)

    @tool
    async def select_monthly_series(version_id: str | None, reason: str) -> str:
        """Select the exact metric and geography from the returned historical-series candidates.

        Select by returned version_id based on the source title and the question. Consider
        narrower regions, rates, private-only measures and transformed series. Return null
        if no candidate fits. This single selection opens the original, verifies the issuer,
        seasonal adjustment, unit and full coverage, and binds each monthly result.
        """
        attempt = adaptive.pending_series_choice()
        if not attempt:
            return json.dumps({"error": "没有待选择的序列候选"}, ensure_ascii=False)
        attempt.update(status="reading", selected_version_id=version_id)
        adaptive.save()
        node_id, part_id, spec = attempt["node_id"], attempt["question_part_id"], attempt["spec"]
        unit = spec["unit"]
        candidate = next((c for c in attempt["candidates"] if c["version_id"] == version_id), None)
        if candidate is None or not reason.strip() or len(reason) > 500:
            return unresolved(attempt, ValueError("未选择符合问题的目录记录"))
        event(
            "monthly_series_source_selected",
            url=candidate["url"],
            title=candidate["title"],
            reason=reason,
        )
        source_url = candidate["url"]
        try:
            budget.before_tool("read")
            doc = await corpus.aread(version_id, timeout=min(30, budget.remaining_time()))
            if fred_path(doc.url)[1] == "series":
                link = history_link(doc)
                if not link:
                    raise ValueError("指标页未提供同一序列的完整数据链接")
                event("monthly_series_history_link", url=link, landing_url=doc.url)
                source_url = link
                version_id = corpus.register(link, doc.title)
                budget.before_tool("read")
                doc = await corpus.aread(version_id, timeout=min(30, budget.remaining_time()))
            if corpus.source_metadata[version_id].get("truncated"):
                raise ValueError("历史表提取被截断")
            source_title = candidate["title"].split(" | ")[0].removeprefix("Table Data - ")
            series_id = fred_path(doc.url)[2]
            source_title = re.sub(rf"\s+\({re.escape(series_id)}\)$", "", source_title, flags=re.I)
            metadata, points = extract_monthly_changes(
                doc, spec, [source_title], attempt["publisher_terms"], attempt["adjustment"]
            )
            evidence.add_document(doc)
            originals = evidence.add_source_spans(
                doc.version_id, (span for point in points for span in point["spans"])
            )
            expose_passages(evidence, budget, list(originals.values()), method="monthly_history")
            result = []
            node = graph.data["nodes"][node_id]
            for index, point in enumerate(points):
                month, value = point["month"], point["value"]
                latest = index == len(points) - 1
                passages = [originals[span] for span in point["spans"]]
                finding = Finding(
                    assertion=f"{month} 环比净变化 {value:+,} {unit}（同一版本相邻月相减）。",
                    occurrence="observed",
                    stance="supports",
                    event_time=TimeSpan(precision="month", start=month),
                    monthly_value=value,
                    monthly_calculation="adjacent_levels",
                    answers_question_part=latest,
                    question_part_id=spec["anchor_part_id"] if latest else part_id,
                    originals=[
                        OriginalRef(passage_id=p.passage_id, quote=p.text) for p in passages
                    ],
                )
                graph.append(node_id, finding, len(node["findings"]))
                result.append(
                    {
                        "month": month,
                        "value": value,
                        "finding_id": node["findings"][-1]["finding_id"],
                        "passage_ids": [p.passage_id for p in passages],
                    }
                )
            receipt = {
                "url": doc.url,
                "version_id": doc.version_id,
                "content_sha256": doc.content_hash,
                "publisher": metadata["Source"],
                "unit": unit,
                "points": result,
            }
            attempt.update(status="completed", **receipt)
            adaptive.refresh()
            event("monthly_series_calculated", **receipt)
            return json.dumps({"status": "completed", **receipt}, ensure_ascii=False)
        except (BudgetExceeded, ValueError, httpx.HTTPError, ArithmeticError) as exc:
            event("monthly_series_source_rejected", url=source_url, reason=str(exc))
            return unresolved(attempt, exc)

    return [collect_monthly_history, select_monthly_series]
