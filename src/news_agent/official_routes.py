"""Recover missing first-party coverage through separate publisher and issuer routes."""

import re
from datetime import timedelta
from urllib.parse import unquote, urlsplit

import httpx

from .retrieval import RetrievalFailure, public_url
from .source_paths import discover_links

RELEASE = re.compile(
    r"now[- ]available|available[- ]now|out[- ]now|has[- ]launched|released[- ]today|现已(?:发售|上市)|本日発売",
    re.I,
)
EARLY = re.compile(
    r"pre[- ]?(?:load|order)|trailer|will (?:launch|release)|set to launch|预载|预告|预约|予約",
    re.I,
)
DISCLOSURE = re.compile(
    r"(?:reports?|announces?|results).{0,65}(?:quarter|fiscal|financial)|(?:quarter|fiscal|financial).{0,70}(?:results|earnings)|earnings release|财务业绩|決算",
    re.I,
)
KANA = re.compile(r"[ぁ-ヿ]|(?:発売|販売本数|売上|決算)")


def folded(text):
    return re.sub(r"[\W_]+", " ", unquote(text).casefold()).strip()


def subject_name(subject):
    # Some planners put the requested measure after the entity in a search-style string.
    cleaned = re.split(
        r"\s+(?:sales units|units sold|copies sold|latest sales|sales figures|sales volume)\b",
        subject,
        maxsplit=1,
        flags=re.I,
    )[0]
    return cleaned.strip() or subject


def language_signals(state, corpus):
    queries = [q for a in state["attempts"] for q in a["queries"].values()]
    hits = {v for a in state["attempts"] for ids in a["hits"].values() for v in ids}
    jp = sum(
        bool(
            KANA.search(corpus.candidates[v].get("title", ""))
            or re.search(r"/(?:jp|ja)(?:[-/]|$)", corpus.candidates[v]["url"])
        )
        for v in hits
        if v in corpus.candidates
    )
    locales = [
        m[1].lower()
        for v in hits
        if v in corpus.candidates
        if (
            m := re.search(
                r"/(ja|jp|zh|de|fr|es|pt|ko|ru|it)(?:[-/]|$)",
                urlsplit(corpus.candidates[v]["url"]).path,
                re.I,
            )
        )
    ]
    return {
        "localized_queries": sum(bool(re.search(r"[^\x00-\x7f]", q)) for q in queries),
        "concentrated_locale": max(set(locales), key=locales.count)
        if locales and len(locales) >= max(2, len(hits) * 0.75)
        else None,
        "queries": len(queries),
        "japanese_hits": jp,
        "hits": len(hits),
    }


def route_score(hit, role, subject):
    title = folded(hit["title"] or "")
    path = folded(urlsplit(hit["url"]).path)
    combined = title + " " + path
    named = folded(subject) in combined
    if role == "latest_update":
        return (
            int(named and bool(RELEASE.search(combined)) and not EARLY.search(combined)),
            int(named and not EARLY.search(combined)),
            int(named),
            0,
        )
    return (
        int(bool(DISCLOSURE.search(combined))),
        int(
            any(
                w in combined
                for w in ("news release", "press release", "earnings", "results", "filing")
            )
        ),
        int(named),
        0,
    )


def collect_product_disclosures(check, subject, domains):
    """Offer the writer traceable product-level observations, with their source and period."""
    from .answer_contract import dates
    from .evidence_chain import expose_passages

    names = {folded(subject_name(subject))}
    alias = check.data.get("original_question", "").split()
    names.update(folded(t) for t in alias if re.fullmatch(r"[a-zA-Z]+[0-9]+", t))
    names.update(re.sub(r"(?<=\D)(?=\d)", " ", name) for name in list(names))
    rows = {}
    for vid in dict.fromkeys(check.data["read_ids"]):
        doc = check.evidence.documents[vid]
        host = (urlsplit(doc.url).hostname or "").lower()
        official = any(host == d or host.endswith("." + d) for d in domains)
        relevant_title = any(" " + name + " " in " " + folded(doc.title) + " " for name in names)
        visible = []
        for m in re.finditer(r"[^\n]+", doc.content):
            text = m[0].strip()
            if "](" in text or not 12 <= len(text) <= 1500:
                continue
            visible.append((m.start(), m.end(), text))
        cutoff_notes = [
            r
            for r in visible
            if re.search(
                r"(based on company estimates|as of|through end of day|data through|统计截至)",
                r[2],
                re.I,
            )
            and len(r[2]) < 350
        ]
        datelines = [
            (a + off, a + end, day)
            for a, _, t in visible
            for day, off, end in dates(t)
            if day <= check.cutoff.isoformat()
            and (
                len(t) < 80
                or re.search(
                    r"(?:NEW YORK|released?|press release|announcement|announced|issued)",
                    t[:100],
                    re.I,
                )
            )
        ]
        for a, b, t in visible:
            if len(rows) >= 65:
                break
            for sentence in re.finditer(r".+?(?:[.!?。](?=\s|$)|$)", t):
                q = sentence[0].strip()
                if not 12 <= len(q) <= 750 or not re.search(
                    r"(units?|copies|pre.?orders?|retail sales|sales revenue|售出|销量|预购|销售额)",
                    q,
                    re.I,
                ):
                    continue
                if not re.search(
                    r"(?:[$£€]\s*\d|\d[\d,.]*\s*(?:million|billion|thousand|units|copies|万|亿|份))",
                    q,
                    re.I,
                ):
                    continue
                named = any(" " + name + " " in " " + folded(q) + " " for name in names)
                if not named and not relevant_title:
                    continue
                if not named and re.search(
                    r"franchise|series|portfolio|net bookings|fiscal (?:year|quarter)|earnings|anticipate|forecast",
                    q,
                    re.I,
                ):
                    continue
                pos = doc.content.find(q, a, b)
                if pos < 0:
                    continue
                context = []
                if datelines:
                    d0, d1, _ = min(datelines, key=lambda r: abs(r[0] - pos))
                    context.append((d0, d1))
                context.extend((x, y) for x, y, _ in cutoff_notes[:1])
                passages = check.evidence.add_source_spans(vid, [(pos, pos + len(q)), *context])
                expose_passages(
                    check.evidence,
                    check.budget,
                    passages.values(),
                    method="product_disclosure_original",
                )
                ps = list(passages.values())
                primary = ps[0]
                rows[primary.passage_id] = {
                    "passage_id": primary.passage_id,
                    "text": q,
                    "source": host,
                    "title": doc.title,
                    "official": official,
                    "publication_day": min(datelines, key=lambda r: abs(r[0] - pos))[2]
                    if datelines
                    else None,
                    "url": doc.url,
                    "basis": [{"id": r.passage_id, "text": r.text} for r in ps[1:]],
                }
    check.data["metric_evidence"] = list(rows.values())
    check.save()


async def recover(
    check,
    subject,
    issuer,
    publisher_index,
    disclosure_index,
    update_query,
    disclosure_query,
    config,
):
    from .answer_contract import dates
    from .current_check import PREORDERS, QUANTITY, RELEASED, _host_allowed
    from .evidence_chain import expose_passages

    subject = subject_name(subject)
    domains = {d for a in check.data["attempts"] for d in a["domains"]}
    pages = [
        ("latest_update", public_url(publisher_index)),
        ("requested_fact", public_url(disclosure_index)),
    ]
    if any(not u.startswith("https://") or not _host_allowed(u, domains) for _, u in pages):
        raise ValueError("Select HTTPS indexes inside the declared official domains.")
    if not all(
        2 <= len(t.strip()) <= 320 for t in (subject, issuer, update_query, disclosure_query)
    ):
        raise ValueError("Provide the canonical subject, issuer and focused search queries.")
    state = check.data.setdefault("official_recovery", {})
    if state.get("finished"):
        return state
    state.update(
        subject=subject,
        issuer=issuer,
        language_signals=language_signals(check.data, check.corpus),
        routes={},
        evidence=[],
    )
    check.save()
    store = check.adaptive.store
    store.event(
        "official_coverage_replanned",
        subject=subject,
        signals=state["language_signals"],
        publisher=publisher_index,
        disclosure=disclosure_index,
    )
    direct = check.data.setdefault("official_pages", {})

    async def load(hit, role, focus):
        vid = hit["version_id"]
        check.corpus.candidates[vid]["discovery_query"] = focus
        if role != "secondary":
            direct[vid] = {"role": role, "focus": focus}
        if vid not in check.data["read_attempted"]:
            check.data["read_attempted"].append(vid)
        import json

        receipt = json.loads(
            await check.reader.ainvoke({"version_id": vid, "passage_limit": 6}, config)
        )
        if not receipt.get("passages"):
            return None
        if vid not in check.data["read_ids"]:
            check.data["read_ids"].append(vid)
        doc = check.evidence.documents[vid]
        check.data["read_ids"] = list(dict.fromkeys(check.data["read_ids"]))
        if role == "latest_update":
            check.observe_release(vid, receipt)
            # Use the dated story itself. A listing or a link to an article is discovery only.
            if not check.data.get("freshness", {}).get("release") and folded(subject) in folded(
                doc.title + " " + urlsplit(doc.url).path
            ):
                for m in RELEASED.finditer(doc.content):
                    a = max(0, m.start() - 240)
                    b = min(len(doc.content), m.end() + 160)
                    nearby = doc.content[a:b]
                    if (
                        folded(subject) not in folded(nearby)
                        or EARLY.search(nearby)
                        or "](" in nearby
                    ):
                        continue
                    choices = [
                        (abs(m.start() - start), day, start, end)
                        for day, start, end in dates(doc.content)
                        if check.cutoff.isoformat()
                        >= day
                        >= (check.cutoff - timedelta(days=45)).isoformat()
                        and abs(start - m.start()) < 1300
                        and "](" not in doc.content[min(start, m.start()) : max(start, m.start())]
                    ]
                    if not choices:
                        continue
                    _, day, start, end = min(choices)
                    ps = check.evidence.add_source_spans(vid, [(a, b), (start, end)])
                    expose_passages(
                        check.evidence,
                        check.budget,
                        ps.values(),
                        method="official_release_original",
                    )
                    refs = check.refs_for([p.passage_id for p in ps.values()])
                    check.data.setdefault("freshness", {})["release"] = {
                        "day": day,
                        "url": doc.url,
                        "references": refs,
                    }
                    store.event(
                        "current_release_confirmed",
                        date=day,
                        url=doc.url,
                        passage_ids=[p.passage_id for p in ps.values()],
                    )
                    break
        chunks = receipt["passages"]
        state["evidence"].append(
            {
                "role": role,
                "title": doc.title,
                "url": doc.url,
                "passages": [{"id": p["passage_id"], "text": p["text"][:700]} for p in chunks[:4]],
            }
        )
        check.save()
        return doc

    async def search(query, domain=None, window=None):
        check.budget.before_tool("search")
        try:
            return await check.corpus.asearch(
                query,
                limit=8,
                include_domains=[domain] if domain else [],
                publication_window=window,
                timeout=min(30, check.budget.remaining_time()),
            )
        except (RetrievalFailure, httpx.HTTPError) as exc:
            state.setdefault("errors", []).append(type(exc).__name__ + ": " + str(exc)[:120])
            return []

    for role, index in pages:
        host = urlsplit(index).hostname.removeprefix("www.")
        queries = (
            [update_query, subject + " official now available launch announcement"]
            if role == "latest_update"
            else [
                disclosure_query,
                issuer + " reports results fiscal quarter " + str(check.cutoff.year),
            ]
        )
        focus = (
            subject
            + " "
            + (
                "release now available"
                if role == "latest_update"
                else "units sold financial results fiscal quarter"
            )
        )
        route = state["routes"][role] = {
            "domain": host,
            "index": index,
            "queries": queries,
            "originals": [],
        }
        hits = {}
        for query in dict.fromkeys(queries):
            for hit in await search(query, host):
                if _host_allowed(hit["url"], [host]):
                    hits[hit["version_id"]] = hit

        def links(doc, focus, role, host, hits):
            for link in discover_links(
                check.evidence,
                check.budget,
                doc.version_id,
                focus,
                limit=6,
                anchor_matches=lambda title: (
                    bool(RELEASE.search(title) and not EARLY.search(title))
                    if role == "latest_update"
                    else bool(DISCLOSURE.search(title))
                ),
            ):
                if _host_allowed(link["url"], [host]):
                    vid = check.corpus.register(link["url"], link["anchor"])
                    hits[vid] = check.corpus.candidates[vid]

        vid = check.corpus.register(index)
        index_doc = await load(check.corpus.candidates[vid], role, focus)
        if index_doc:
            links(index_doc, focus, role, host, hits)
        # A publisher's front page can feature an announcement before the newsroom/search index updates.
        if role == "latest_update" and urlsplit(index).path.strip("/"):
            root = check.corpus.register("https://" + urlsplit(index).netloc + "/")
            home = await load(check.corpus.candidates[root], role, focus)
            if home:
                links(home, focus, role, host, hits)
        if not any(route_score(h, role, subject)[0] for h in hits.values()) and hasattr(
            check.corpus, "map_index"
        ):
            try:
                check.budget.before_tool("search")
                for url in await check.corpus.map_index(
                    index, focus, min(30, check.budget.remaining_time())
                ):
                    if not _host_allowed(url, [host]):
                        continue
                    vid = check.corpus.register(url)
                    hits[vid] = check.corpus.candidates[vid]
            except (RetrievalFailure, ValueError, httpx.HTTPError) as exc:
                route["map_error"] = str(exc)[:120]
        if role == "requested_fact":
            parent = next(
                (d for d in sorted(domains, key=len) if host == d or host.endswith("." + d)), host
            )
            # Follow a recent product disclosure even when a different host syndicated it.
            # The headline is a lead. Only a read original on an allowed domain is evidence.
            recent = ((check.cutoff - timedelta(days=14)).isoformat(), check.cutoff.isoformat())
            alias = re.search(r"[a-zA-Z]+\s*\d+", check.data.get("original_question", ""))
            search_name = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", alias[0]) if alias else subject
            compact = search_name + " units sold copies shipped"
            discovery = await search(compact, window=recent)
            headlines = [
                h["title"].split("|")[0].strip()
                for h in discovery
                if (
                    folded(subject) in folded(h["title"])
                    or folded(search_name) in folded(h["title"])
                )
                and re.search(r"sales|sold|records|milestone|units|copies", h["title"], re.I)
            ]
            for query in dict.fromkeys([compact, *(headlines[:1])]):
                for h in await search(query, parent):
                    if _host_allowed(h["url"], [parent]):
                        hits[h["version_id"]] = h
            for h in [*discovery, *check.corpus.candidates.values()]:
                if _host_allowed(h["url"], [parent]) and folded(subject) in folded(h["title"]):
                    hits[h["version_id"]] = h
            route["queries"].extend([compact, *headlines[:1]])
        ranked = sorted(
            hits.values(),
            key=lambda h: (
                (
                    int(
                        folded(subject) in folded(h["title"])
                        and bool(re.search(r"sales|sold|records|milestone", h["title"], re.I))
                    )
                    if role == "requested_fact"
                    else 0
                ),
                route_score(h, role, subject),
                h.get("provider_published_date") or "",
            ),
            reverse=True,
        )
        seen = set()
        for hit in ranked:
            u = hit["url"].split("?")[0].rstrip("/")
            if u == index.rstrip("/") or u in seen:
                continue
            seen.add(u)
            doc = await load(hit, role, focus)
            if doc:
                route["originals"].append(
                    {"id": doc.version_id, "url": doc.url, "title": doc.title}
                )
            if role == "latest_update" and check.data.get("freshness", {}).get("release"):
                break
            if len(seen) >= 3:
                break
        store.event(
            "official_route_checked",
            role=role,
            domain=host,
            originals=route["originals"],
            index_read=bool(index_doc),
        )
        check.save()
    if check.sales_question:
        q = (
            subject
            + " estimated preorders number of copies preorders revenue analyst estimates "
            + str(check.cutoff.year)
        )
        hits = await search(q)
        third = [h for h in hits if not _host_allowed(h["url"], domains)]
        third.sort(
            key=lambda h: (
                bool(PREORDERS.search(h["title"]) and QUANTITY.search(h["title"])),
                bool(PREORDERS.search(h["title"] + " " + h.get("snippet", ""))),
            ),
            reverse=True,
        )
        for h in third[:2]:
            doc = await load(h, "secondary", subject + " estimated preorders copies units revenue")
            if not doc:
                continue
            ps = {}
            for m in PREORDERS.finditer(doc.content):
                a = max(
                    0,
                    doc.content.rfind("\n", max(0, m.start() - 350), m.start()) + 1,
                    m.start() - 350,
                )
                end = doc.content.find("\n", m.end())
                b = min(len(doc.content), end if end != -1 else len(doc.content), m.end() + 450)
                if QUANTITY.search(doc.content[a:b]) and "](" not in doc.content[a:b]:
                    ps.update(check.evidence.add_source_spans(doc.version_id, [(a, b)]))
                if len(ps) >= 3:
                    break
            if ps:
                expose_passages(
                    check.evidence, check.budget, ps.values(), method="attributed_estimate_original"
                )
                state.setdefault("secondary_refs", []).extend(
                    {**r, "source_url": doc.url}
                    for r in check.refs_for([p.passage_id for p in ps.values()])
                )
                break
    collect_product_disclosures(check, subject, domains)
    state["finished"] = True
    check.save()
    return state
