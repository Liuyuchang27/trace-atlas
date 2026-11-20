"""Keep discovery routes available; links remain leads until the agent reads them."""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace
from urllib.parse import urljoin

from langchain_core.tools import tool

from .budget import BudgetExceeded
from .evidence_chain import rank_passages
from .retrieval import public_url


def discover_links(evidence, budget, version_id, query, limit=4, *, anchor_matches=None):
    doc = evidence.documents[version_id]
    rows = []
    for match in re.finditer(r"\[([^\]\n]{1,300})\]\(([^\s)]+)\)", doc.content):
        if anchor_matches and not anchor_matches(match[1]):
            continue
        try:
            url = public_url(urljoin(doc.url, match[2]))
        except ValueError:
            continue
        if url == public_url(doc.url):
            continue  # page-local contents links are not new source paths
        # Bound the immediate paragraph for ranking only; do not show it as evidence.
        previous_break = doc.content.rfind("\n\n", 0, match.start())
        start = max(previous_break + 2 if previous_break >= 0 else 0, match.start() - 200)
        end = doc.content.find("\n\n", match.end())
        end = min(end if end >= 0 else len(doc.content), match.end() + 200)
        rows.append(
            SimpleNamespace(
                text=match[1] + " " + doc.content[start:end],
                start=match.start(),
                end=match.end(),
                url=url,
                anchor=match[1],
                original=match[0],
            )
        )
    selected, seen = [], set()
    for ranked in rank_passages(rows, query or doc.title):
        row = ranked["passage"]
        if row.url in seen or not ranked["matched_terms"]:
            continue
        seen.add(row.url)
        identity = f"{version_id}:{row.start}:{row.url}"
        selected.append(
            {
                "link_id": "link-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                "from_version_id": version_id,
                "anchor": row.anchor,
                "url": row.url,
                "start": row.start,
                "end": row.end,
                "original_link": row.original,
            }
        )
        if len(selected) >= max(1, min(limit, 6)):
            break
    leads = budget.state.setdefault("source_leads", {})
    for row in selected:
        leads.pop(row["link_id"], None)
        leads[row["link_id"]] = row
    budget.state["source_leads"] = dict(list(leads.items())[-24:])
    budget.save()
    evidence.store.event(
        "source_links_discovered",
        version_id=version_id,
        query=query,
        links=selected,
        scope="exact saved link targets; discovery only, not source verification",
    )
    return selected


def source_link_tool(evidence, budget):
    @tool
    async def list_source_links(version_id: str, query: str, limit: int = 4) -> str:
        """Find relevant outbound links in your selected saved page; no network request.

        Links are discovery leads, not original evidence from the target. Use open_url on a
        chosen URL to read it. Use keywords for a missing original, dated decision or new case.
        """
        try:
            budget.before_tool("read")
            if version_id not in evidence.documents or not query.strip():
                return json.dumps({"error": "read the source page first and provide keywords"})
            return json.dumps(
                {
                    "links": discover_links(evidence, budget, version_id, query, limit),
                    "scope": "leads only; choose open_url to read target originals",
                }
            )
        except BudgetExceeded as exc:
            return json.dumps({"error": str(exc)})

    return list_source_links


def pending_routes(evidence, budget):
    """Mix earlier and recent unvisited candidates, retaining their own discovery context."""
    candidates = list(budget.state.get("search_candidates", {}).values())
    unread = [h for h in candidates if h["version_id"] not in evidence.documents]
    # Avoid assigning trust by domain. The model sees title/snippet/URL and chooses the route.
    latest = {h["version_id"] for h in budget.state.get("last_search_results", [])}
    older = [h for h in unread if h["version_id"] not in latest]
    recent = [h for h in unread if h["version_id"] in latest]
    order = [*recent[:2], *older[-3:], *recent[2:], *older[:-3]]
    selected = list({h["version_id"]: h for h in order}.values())[:5]
    visited = {d.url for d in evidence.documents.values()}
    links = [r for r in budget.state.get("source_leads", {}).values() if r["url"] not in visited]
    return (
        [
            {
                "version_id": h["version_id"],
                "title": h["title"][:110],
                "url": h["url"],
                "snippet": h.get("snippet", "")[:160],
            }
            for h in selected
        ],
        [{k: r[k] for k in ("link_id", "from_version_id", "anchor", "url")} for r in links[-4:]],
    )
