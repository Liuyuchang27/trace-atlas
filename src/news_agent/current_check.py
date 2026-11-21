"""Bounded verification of one explicitly current fact against its first-party sources."""

import json
import re
from datetime import date, datetime, timedelta
from typing import Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .answer_contract import dates
from .budget import BudgetExceeded
from .current_report import answer_lead, bound_points, synthesize_notes
from .event_graph import OriginalRef
from .official_routes import language_signals, recover
from .retrieval import RetrievalFailure, public_url, publication_day
from .schemas import StrictModel
from .source_paths import discover_links

MARKER = "\n\nRun observation context: "
LABELS = {
    "confirmed": "已查到一手来源",
    "not_found": "在已核查的一手来源中未找到",
    "insufficient": "证据不足，无法确认",
}
INSUFFICIENT_DETAIL = "本轮未取得能确认所问状态或数据的原文。"
PRELAUNCH = re.compile(
    r"\bpre[-\s]?(?:load(?:ing)?|order(?:s|ing)?)\b|\bcoming soon\b|"
    r"\bwill (?:launch|be released)\b|预载|预购|预售",
    re.I,
)
PREORDERS = re.compile(r"\bpre[-\s]?order(?:s|ing)?\b|预购|预售", re.I)
QUANTITY = re.compile(r"\d[\d,.]*\s*(?:million|billion|copies|units|万|亿|份)", re.I)
ESTIMATE = re.compile(r"\bestimat\w*\b|预估|估算|估计", re.I)
SALES_QUESTION = re.compile(r"销量|销售量|售出|\bsales\b|\b(?:units|copies) sold\b", re.I)
RELEASED = re.compile(
    r"\b(?:now available|available now|out now|has launched|released today|is now live)\b(?!\s+for\s+pre)|"
    r"现已(?:发售|上市|推出)|已正式(?:发售|上市)",
    re.I,
)
UNSUPPORTED_PRELAUNCH = re.compile(
    r"(?:尚未|还没|并未|未)(?:正式)?(?:发售|上市)|处于(?:发售前|预购|预载)|"
    r"\b(?:has not (?:yet )?(?:launched|been released)|is not yet available|pre-launch phase)\b",
    re.I,
)
UNKNOWN_LAUNCH_ANSWER = (
    "已读取的预载或预购记录只能说明当时的状态。发售状态待确认，所查来源未提供可核实的实际销量。"
)


class OfficialPage(StrictModel):
    url: str
    role: Literal["latest_update", "requested_fact"]
    focus: str


def applies(task):
    question = re.sub(r"https?://[^\s，。；]+", "", task.question.split(MARKER, 1)[0])
    return (
        task.cutoff is None
        and MARKER in task.question
        and len(question) <= 100
        and re.search(
            r"最新|目前|现在|截至今天|\blatest\b|\bcurrent(?:ly)?\b|\bas of today\b", question, re.I
        )
        and not re.search(
            r"历程|趋势|对比|比较|逐月|过去|近[一二三四五六七八九十半\d]|复盘|时间线|沿革|"
            r"截至\s*\d{4}|\btrend\b|\bhistory\b|\bcompare\b|\bas of \d{4}\b|上一请求",
            question,
            re.I,
        )
    )


def _host_allowed(url, domains):
    host = (urlsplit(url).hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def previous_read_candidates(hits, domains):
    """Reserve one original disclosure and one relevant attributed report when available."""
    primary = [h for h in hits if _host_allowed(h["url"], domains)]
    reporting = [h for h in hits if not _host_allowed(h["url"], domains)]
    reporting.sort(
        key=lambda h: (
            bool(QUANTITY.search(h["title"]) and PREORDERS.search(h["title"])),
            bool(PREORDERS.search(h["title"])),
        ),
        reverse=True,
    )
    return [*primary[:1], reporting[0]] if reporting else primary[:2]


def checked_sources(state, documents, passages):
    """Only saved original text from pages actually read can appear in the report."""
    citations = {}
    launch = state.get("freshness", {}).get("release") or {}
    refs = (
        (state.get("result") or {}).get("references", [])
        + launch.get("references", [])
        + state.get("previous_period", {}).get("related_refs", [])
    )
    for row in refs:
        citations.setdefault(row["version_id"], []).append(row["passage_id"])
    return [
        {
            "version_id": vid,
            "url": doc["url"],
            "title": doc["title"],
            "cited": list(dict.fromkeys(citations.get(vid) or passes[:1])),
        }
        for vid in state.get("read_ids", [])
        if (doc := documents.get(vid))
        and (passes := [pid for pid, p in passages.items() if p["version_id"] == vid])
    ]


def result_view(state):
    result = state.get("result") or {"outcome": "insufficient", "text": "", "record_on": None}
    freshness = state.get("freshness", {})
    release = freshness.get("release")
    stale = bool(freshness.get("stale_sources"))
    if stale and not release and result["outcome"] != "confirmed":
        result = {
            **result,
            "outcome": "insufficient",
            "record_on": None,
            "text": UNKNOWN_LAUNCH_ANSWER,
        }
    if state.get("sales_question") and result["outcome"] != "confirmed":
        subject = state.get("official_recovery", {}).get("subject", "该产品")
        result = {
            **result,
            "text": f"截至本次核查，所读官方原文中未找到 {subject} 截至核查时点的累计售出总份数。",
        }
    older_record = result.get("data_through") or result.get("record_on")
    label = (
        "最近一次可核实的实际数据"
        if result["outcome"] == "confirmed" and older_record and older_record < state["cutoff"]
        else LABELS[result["outcome"]]
    )
    return {
        **result,
        "text": result["text"] or INSUFFICIENT_DETAIL,
        "label": label,
        "cutoff": state["cutoff"],
        "cutoff_label": state.get("cutoff_label", state["cutoff"]),
        **({"release": release} if release else {}),
        "release_pending": stale and not release,
    }


def written_view(state, draft):
    view = result_view(state)
    points = bound_points(state, draft)
    lead = answer_lead(state, draft)
    if lead:
        view["text"] = lead["text"]
    if points:
        best = next(
            (
                p
                for p in points
                if p["official"] and re.search(r"份|套|units|copies", p["summary"], re.I)
            ),
            next((p for p in points if p["official"]), points[0]),
        )
        rest = [p for p in points if p is not best]
        qualifier = (lead or {}).get("text", "") if best["scope"] != "direct" else ""
        if state.get("sales_question") and best["scope"] != "direct" and not qualifier:
            qualifier = "截至核查时点，所读原文未能确认发售后的累计销量。"
        view["text"] = best["summary"] + " " + qualifier
        view["lead_ids"] = best["citations"]
        view["lead_source"] = best["source"]
        view["lead_published"] = best.get("publication_day")
        view["reported_scope"] = best["scope"]
        view["other_findings"] = rest
        view["lead_official"] = best["official"]
    return view


def related_preorder_notes(state):
    """Keep earlier preorder observations without upgrading them into actual sales."""
    notes = {}
    previous = state.get("previous_period", {})
    recovery = state.get("official_recovery", {})
    secondary_refs = recovery.get("secondary_refs", [])
    secondary = (set(previous.get("hits", [])) - set(previous.get("official_ids", []))) | {
        r["version_id"] for r in secondary_refs
    }

    refs = (state.get("result") or {}).get("references", []) + state.get("previous_period", {}).get(
        "related_refs", []
    )
    for ref in [*secondary_refs, *refs]:
        quote = ref.get("quote", "")
        # Use complete statements, so a later qualifier is not lost by truncating it.
        for line in re.split(r"(?<=[.!?。！？])\s+|\n+|(?<=。)", quote):
            text = line.strip().removeprefix("- ")
            if not (12 <= len(text) <= 650 and PREORDERS.search(text) and QUANTITY.search(text)):
                continue
            if re.search(
                r"net (?:bookings|income)|guidance|historic(?:al)? year|财年指引|历史性的一年",
                text,
                re.I,
            ):
                continue
            if text.endswith((".", "。", "!", "！", "?", "？")):
                prior = notes.get(ref["version_id"])
                if not prior or (QUANTITY.search(text) and not QUANTITY.search(prior["text"])):
                    label = "预购记载"
                    if ref["version_id"] in secondary:
                        label = "第三方预购估算" if ESTIMATE.search(text) else "第三方预购报道"
                    notes[ref["version_id"]] = {
                        "text": text,
                        "source": next(
                            (
                                row["url"]
                                for row in recovery.get("evidence", [])
                                if any(
                                    p["id"] == ref["passage_id"] for p in row.get("passages", [])
                                )
                            ),
                            ref.get("source_url", ""),
                        ),
                        "label": label,
                        "passage_id": ref["passage_id"],
                        "version_id": ref["version_id"],
                    }
    return sorted(notes.values(), key=lambda n: bool(QUANTITY.search(n["text"])), reverse=True)[:2]


def report_preorder_notes(state, draft):
    points = [p for p in bound_points(state, draft) if not p["official"]]
    covered_urls = {p["url"] for p in points}
    covered_ids = {p["passage_id"] for p in points}
    versions = {
        r["version_id"]
        for r in state.get("official_recovery", {}).get("secondary_refs", [])
        if r.get("source_url") in covered_urls or r.get("passage_id") in covered_ids
    }
    return [
        n
        for n in synthesize_notes(related_preorder_notes(state), draft)
        if n.get("source") not in covered_urls and n["version_id"] not in versions
    ]


class CurrentCheck:
    def __init__(self, task, adaptive, corpus, reader):
        observed = json.loads(task.question.split(MARKER, 1)[1])
        zone = observed["timezone"]
        clock = datetime.fromisoformat(observed["local"]).astimezone(ZoneInfo(zone))
        self.cutoff = clock.date()
        self.sales_question = bool(SALES_QUESTION.search(task.question.split(MARKER, 1)[0]))
        self.release_sensitive = bool(
            re.search(
                r"销量|售出|发售|预载|预购|\bsales\b|\bcopies sold\b|\brelease status\b|\blaunch\b",
                task.question.split(MARKER, 1)[0],
                re.I,
            )
        )
        timezone_label = "香港时间" if zone == "Asia/Hong_Kong" else zone
        self.adaptive, self.corpus, self.reader = adaptive, corpus, reader
        self.budget, self.evidence = adaptive.budget, adaptive.evidence
        self.data = adaptive.data.setdefault(
            "current_check",
            {
                "cutoff": self.cutoff.isoformat(),
                "cutoff_label": f"{clock:%Y-%m-%d %H:%M}（{timezone_label}）",
                "turns": 0,
                "attempts": [],
                "read_ids": [],
                "read_attempted": [],
            },
        )
        self.data["original_question"] = task.question.split(MARKER, 1)[0].strip()
        self.data["sales_question"] = self.sales_question
        adaptive.save()

    def save(self):
        self.adaptive.save()

    def observe_release(self, vid, receipt):
        if not self.release_sensitive or not receipt.get("passages"):
            return
        doc = self.evidence.documents[vid]
        title = doc.title.split(" | ")[0]
        current = self.data.setdefault("freshness", {})
        first = doc.content[:8000]
        subject = self.data.get("official_recovery", {}).get("subject")
        if (
            (not subject or subject.casefold() in title.casefold())
            and RELEASED.search(title)
            and not PRELAUNCH.search(title)
            and not UNSUPPORTED_PRELAUNCH.search(title)
        ):
            # An announcement saying it is available now establishes availability
            # at its own visible date. Index titles and retrieval dates are excluded.
            passages = sorted(receipt["passages"], key=lambda row: row["start"])
            proof = next(
                ((row, match) for row in passages if (match := RELEASED.search(row["text"]))), None
            )
            date_refs = []
            if proof:
                position = proof[0]["start"] + proof[1].start()
                for row in passages:
                    for day, offset, _ in dates(row["text"]):
                        distance = position - row["start"] - offset
                        if (
                            abs(distance) <= 1800
                            and self.cutoff - timedelta(days=30)
                            <= date.fromisoformat(day)
                            <= self.cutoff
                        ):
                            date_refs.append((abs(distance), distance < 0, day, row["passage_id"]))
            if date_refs:
                _, _, day, date_pid = min(date_refs)
                positive = proof[0]["passage_id"]
                existing = current.get("release")
                if not existing or day > existing["day"]:
                    refs = self.refs_for([positive, date_pid])
                    current["release"] = {"day": day, "url": doc.url, "references": refs}
                    self.adaptive.store.event(
                        "current_release_confirmed",
                        date=day,
                        url=doc.url,
                        passage_ids=[positive, date_pid],
                    )
                return
        if PRELAUNCH.search(title) or PRELAUNCH.search(first):
            stale = current.setdefault("stale_sources", [])
            if vid not in stale:
                stale.append(vid)
                self.adaptive.store.event("current_source_prelaunch", version_id=vid, url=doc.url)

    def refs_for(self, ids):
        passages = [self.evidence.passages[pid] for pid in dict.fromkeys(ids)]
        return self.adaptive.graph._references(
            [OriginalRef(passage_id=p.passage_id, quote=p.text[:900]) for p in passages]
        )

    def begin_previous_period(self):
        if self.sales_question and self.data["attempts"] and "previous_period" not in self.data:
            self.data["previous_period"] = {"stage": "search", "turns": 0}
            self.save()
            return True
        return False

    def prompt(self):
        previous = self.data.get("previous_period")
        checking_previous = previous is not None and previous["stage"] != "done"
        if checking_previous and self.budget.state["phase"] == "research":
            previous["turns"] += 1
            if previous["turns"] > 4:
                previous["stage"] = "done"
                self.budget.finish_research("current_check_incomplete")
            self.save()
        elif self.data.get("result"):
            self.budget.finish_research("current_fact_checked")
        elif self.budget.state["phase"] == "research":
            self.data["turns"] += 1
            self.save()
            if self.data["turns"] > 8 and not self.begin_previous_period():
                self.budget.finish_research("current_check_incomplete")
        return "CURRENT_FACT_CHECK:\n" + json.dumps(
            {
                **self.data,
                "instruction": (
                    "Name the actual measured quantity first. A broad corporate outlook, projected bookings, "
                    "executive enthusiasm or an adjective supplies no product units. Only report background "
                    "when it quantifies the requested product and says whose estimate it is. "
                    "When covering different authorities choose each authority's editorial language "
                    "and distinct host. For an English-language publisher use its English product name and "
                    "English search terms. Recover missing roles using the dedicated coverage tool. "
                    "Answer the user's current fact directly as of cutoff. Start from the "
                    "entity's first-party publisher, regulator or data issuer, with separate "
                    "queries for the latest relevant update and for the exact requested fact. "
                    "Use the official publisher's language. For a product metric, look for "
                    "the official release or availability in the update query, and use a "
                    "separate query for actual sales. Read relevant originals from both lines "
                    "of search and each requested authority when available. "
                    "When search is empty, stale, or misses an authority, use official_pages "
                    "in read_current_sources to inspect the publisher's homepage/newsroom and "
                    "the issuer's investor-relations landing page directly. Follow their relevant "
                    "dated article links. Leave read capacity for those newest records. "
                    "An old pre-load or pre-order page never establishes present unavailability. "
                    "If the freshness state detects one, open the publisher's current official "
                    "homepage and investor-relations portal with refresh_current_status. Use "
                    "the full product name in the publisher's language to find updated notices. "
                    "A plan, estimate, different period, entity or metric is contextual only. "
                    "If the current value is unavailable, check the last reporting period "
                    "for an actual comparable value. Use the most recent verified disclosure "
                    "and give its data cutoff separately when the original specifies one. "
                    "Report the most recent matching dated disclosure, or state what the checked "
                    "sources do not establish. Do not use the cutoff as a fact's own date."
                ),
            },
            ensure_ascii=False,
        )

    def surface(self, tools):
        attempts = self.data["attempts"]
        freshness = self.data.get("freshness", {})
        previous = self.data.get("previous_period", {}).get("stage")
        recovery = self.data.get("official_recovery", {})
        signals = (
            language_signals(self.data, self.corpus)
            if attempts and hasattr(self.corpus, "candidates")
            else {}
        )
        if (
            attempts
            and not recovery.get("finished")
            and (
                freshness.get("stale_sources")
                or (
                    self.release_sensitive
                    and (signals.get("localized_queries") or signals.get("concentrated_locale"))
                )
                or (
                    self.sales_question
                    and (self.data["read_attempted"] or not any(attempts[0]["hits"].values()))
                )
            )
        ):
            return [t for t in tools if t.name == "recover_official_sources"]
        if recovery.get("finished") and not previous:
            return [t for t in tools if t.name == "record_current_result"]
        if previous in {"search", "assess"}:
            names = {"search_previous_sales" if previous == "search" else "record_current_result"}
        elif (
            freshness.get("stale_sources")
            and not freshness.get("release")
            and not freshness.get("attempted")
        ):
            names = {"refresh_current_status"}
        elif set(freshness.get("candidates", [])) - set(
            self.data["read_attempted"]
        ) and not freshness.get("article_attempted"):
            names = {"read_current_sources"}
        elif not attempts:
            names = {"search_current_sources"}
        else:
            names = {"record_current_result"}
            found = {vid for a in attempts for ids in a["hits"].values() for vid in ids}
            unread = found - set(self.data["read_attempted"])
            if len(self.data["read_attempted"]) < 4:
                names.add("read_current_sources")
            if len(attempts) < 2:
                names.add("search_current_sources")
                if not unread and not self.data["read_ids"]:
                    names.discard("record_current_result")
            if "read_current_sources" in names and not self.data["read_attempted"]:
                names.discard("record_current_result")
        return [t for t in tools if t.name in names]

    def tools(self):
        @tool
        async def recover_official_sources(
            subject: str,
            issuer: str,
            publisher_index: str,
            disclosure_index: str,
            update_query: str,
            disclosure_query: str,
            config: RunnableConfig,
        ) -> str:
            """Correct missing or stale official coverage. Give the canonical full product/subject name,
            corporate issuer name, official publisher newsroom URL and financial disclosures/IR URL
            within the declared authorities. Write a fresh release query and a company results query
            in the respective issuer's primary language. Prefer the specific requested product and measure over corporate forecasts or general quarterly results. This reads separate originals and checks for
            a quantified third-party preorder estimate when researching sales."""
            try:
                result = await recover(
                    self,
                    subject,
                    issuer,
                    publisher_index,
                    disclosure_index,
                    update_query,
                    disclosure_query,
                    config,
                )
            except ValueError as exc:
                return json.dumps({"error": str(exc)})
            if self.sales_question and self.data.get("metric_evidence"):
                self.data["result"] = {
                    "outcome": "insufficient",
                    "text": "本轮已取得有统计口径的原文。",
                    "references": [],
                    "record_on": None,
                }
                self.save()
                self.budget.finish_research("current_sources_checked")
            return json.dumps(result, ensure_ascii=False)

        @tool
        async def search_previous_sales(query: str, config: RunnableConfig) -> str:
            """Find the latest documented actual sales from an earlier reporting period.

            Use the same product, scope and metric. Search for actual reported units and the
            relevant reporting period. Issuer domains are preferred. Other sources can supply
            attributed background or leads and do not establish an official figure.
            The tool reads up to two returned candidates and returns their original passages.
            """
            previous = self.data.get("previous_period", {})
            if previous.get("stage") != "search" or not 3 <= len(query.strip()) <= 420:
                return json.dumps(
                    {"error": "Give one focused query for the prior reporting period."}
                )
            since = (self.cutoff - timedelta(days=365)).isoformat()
            domains = sorted({d for attempt in self.data["attempts"] for d in attempt["domains"]})[
                :3
            ]
            previous.update(stage="done", query=query, **{"from": since}, hits=[])
            self.save()
            try:
                self.budget.before_tool("search")
                hits = await self.corpus.asearch(
                    query,
                    limit=8,
                    include_domains=domains,
                    prefer_domains=True,
                    publication_window=(since, self.cutoff.isoformat()),
                    timeout=min(30, self.budget.remaining_time()),
                )
                unread = [
                    hit for hit in hits if hit["version_id"] not in self.data["read_attempted"]
                ]
                unread.sort(
                    key=lambda hit: (
                        _host_allowed(hit["url"], domains),
                        publication_day(hit.get("provider_published_date")) or date.min,
                    ),
                    reverse=True,
                )
                previous.update(
                    hits=[h["version_id"] for h in unread],
                    official_ids=[
                        h["version_id"] for h in unread if _host_allowed(h["url"], domains)
                    ],
                )
                selected = previous_read_candidates(unread, domains)
                previous["selected"] = [h["version_id"] for h in selected]
                self.save()
                originals = []
                for candidate in selected:
                    vid = candidate["version_id"]
                    self.data["read_attempted"].append(vid)
                    self.save()
                    receipt = json.loads(await self.reader.ainvoke({"version_id": vid}, config))
                    if passages := receipt.get("passages"):
                        self.data["read_ids"].append(vid)
                        previous["stage"] = "assess"
                        notes = [
                            row["passage_id"] for row in passages if PREORDERS.search(row["text"])
                        ]
                        if notes:
                            previous.setdefault("related_refs", []).extend(self.refs_for(notes[:2]))
                    originals.append(receipt)
                    self.save()
                return json.dumps(
                    {
                        "candidates": unread,
                        "official_ids": previous["official_ids"],
                        "originals": originals,
                    },
                    ensure_ascii=False,
                )
            except (RetrievalFailure, BudgetExceeded, httpx.HTTPError) as exc:
                previous["error"] = type(exc).__name__
                self.save()
                return json.dumps({"error": previous["error"]})

        @tool
        async def refresh_current_status(
            official_pages: list[OfficialPage], config: RunnableConfig
        ) -> str:
            """Replace pre-launch discovery with a direct read of official publication indexes.

            Choose one or two official HTTPS landing pages, such as the publisher's home
            and investor-relations home. Use the full product name in the official source
            language as focus. The tool returns current release links for you to open.
            """
            freshness = self.data.get("freshness", {})
            domains = {d for attempt in self.data["attempts"] for d in attempt["domains"]}
            try:
                urls = [public_url(page.url) for page in official_pages]
                if (
                    not freshness.get("stale_sources")
                    or freshness.get("attempted")
                    or not 1 <= len(set(urls)) == len(urls) <= 2
                    or any(
                        not url.startswith("https://") or not _host_allowed(url, domains)
                        for url in urls
                    )
                    or any(not 1 <= len(p.focus.strip()) <= 320 for p in official_pages)
                ):
                    raise ValueError(
                        "Choose one or two official HTTPS publication indexes for the stale record."
                    )
            except ValueError as exc:
                return json.dumps({"error": str(exc)})
            freshness.update(attempted=True, candidates=[])
            self.save()
            direct = self.data.setdefault("official_pages", {})
            details = []
            for page, url in zip(official_pages, urls, strict=True):
                vid = self.corpus.register_origin(url, domains, query=page.focus)
                receipt = json.loads(await self.reader.ainvoke({"version_id": vid}, config))
                leads = []
                if receipt.get("passages"):
                    self.data["read_ids"].append(vid)
                    self.data["read_attempted"].append(vid)
                    for link in discover_links(
                        self.evidence,
                        self.budget,
                        vid,
                        page.focus,
                        limit=6,
                        anchor_matches=lambda title: (
                            RELEASED.search(title) and not PRELAUNCH.search(title)
                        ),
                    ):
                        target_url, title = link["url"], link["anchor"]
                        if not target_url.startswith("https://") or not _host_allowed(
                            target_url, domains
                        ):
                            continue
                        target = self.corpus.register_origin(target_url, domains, title, page.focus)
                        if target not in freshness["candidates"]:
                            freshness["candidates"].append(target)
                        direct[target] = {"role": page.role, "focus": page.focus}
                        leads.append({"version_id": target, "url": target_url, "title": title})
                details.append({"url": url, "release_links": leads, "error": receipt.get("error")})
                self.save()
            self.adaptive.store.event("current_status_refreshed", indexes=details)
            return json.dumps({"official_index_results": details}, ensure_ascii=False)

        @tool
        async def search_current_sources(
            latest_update_query: str, requested_fact_query: str, authority_domains: list[str]
        ) -> str:
            """Search recent first-party publications for the current status and exact fact.

            Use source-language queries: latest_update for the newest official status or
            announcement (for example, availability), requested_fact for the exact metric.
            Supply 1-3 actual issuer/publisher/regulator hostnames without a scheme.
            A second search pass uses a wider publication window and requires a changed route.
            """
            attempts = self.data["attempts"]
            queries = {"latest_update": latest_update_query, "requested_fact": requested_fact_query}
            domains = sorted({d.strip().lower() for d in authority_domains})
            normalized = [" ".join(q.casefold().split()) for q in queries.values()]
            if (
                self.data.get("result")
                or len(attempts) >= 2
                or len(set(normalized)) != 2
                or not all(3 <= len(q) <= 500 for q in queries.values())
                or not 1 <= len(domains) <= 3
                or any(not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", d) for d in domains)
            ):
                return json.dumps(
                    {"error": "Choose two distinct queries and 1–3 official hostnames."}
                )
            if attempts and (
                domains == attempts[-1]["domains"]
                and any(q in attempts[-1]["queries"].values() for q in queries.values())
            ):
                return json.dumps({"error": "Change the search queries or add a different issuer."})
            since = self.cutoff - timedelta(days=30 if not attempts else 90)
            update_since = self.cutoff - timedelta(days=7 if not attempts else 30)
            row = {
                "queries": queries,
                "domains": domains,
                "from": since.isoformat(),
                "latest_update_from": update_since.isoformat(),
                "hits": {},
                "errors": {},
            }
            attempts.append(row)
            self.save()
            sources = {}
            for role, query in queries.items():
                try:
                    self.budget.before_tool("search")
                    latest_update = role == "latest_update"
                    hits = await self.corpus.asearch(
                        query,
                        limit=6,
                        timeout=min(30, self.budget.remaining_time()),
                        include_domains=domains,
                        publication_window=(
                            row["latest_update_from"] if latest_update else row["from"],
                            self.cutoff.isoformat(),
                        ),
                        news=latest_update,
                    )
                    hits = [h for h in hits if _host_allowed(h["url"], domains)]
                    sources[role] = hits
                    row["hits"][role] = [hit["version_id"] for hit in hits]
                except (RetrievalFailure, BudgetExceeded, httpx.HTTPError) as exc:
                    row["errors"][role] = type(exc).__name__
                self.save()
            return json.dumps(
                {
                    "publication_windows": {
                        "latest_update": [row["latest_update_from"], self.cutoff.isoformat()],
                        "requested_fact": [row["from"], self.cutoff.isoformat()],
                    },
                    "sources": sources,
                    "errors": row["errors"],
                },
                ensure_ascii=False,
            )

        @tool
        async def read_current_sources(
            config: RunnableConfig,
            version_ids: list[str] | None = None,
            official_pages: list[OfficialPage] | None = None,
        ) -> str:
            """Read current-search candidates, or go directly to a known official publisher.

            Use official_pages for up to two official homepage/newsroom or investor-relations
            landing pages when search misses an authority or returns old material. This opens
            those pages and discovers relevant first-party release links for the next read.
            Supply official_pages by themselves. Tag status pages latest_update and issuer
            disclosure pages requested_fact. Four total page reads; reserve two for release links.
            """
            attempts = self.data["attempts"]
            direct = self.data.setdefault("official_pages", {})
            domains = {domain for a in attempts for domain in a["domains"]}
            candidates = {
                vid for a in attempts for ids in a["hits"].values() for vid in ids
            } | direct.keys()
            seen = self.data["read_attempted"]
            ids = list(dict.fromkeys(version_ids or []))
            pages = official_pages or []
            freshness = self.data.get("freshness", {})
            fresh = set(freshness.get("candidates", [])) - set(seen)
            followup = bool(fresh and not freshness.get("article_attempted"))
            try:
                urls = [public_url(page.url) for page in pages]
                if (
                    not 1 <= len(ids) + len(pages) <= (2 if followup else 4 - len(seen))
                    or len(pages) > 2
                    or (pages and ids)
                    or (followup and (pages or set(ids) - fresh))
                    or len(set(urls)) != len(urls)
                    or set(ids) - candidates
                    or set(ids) & set(seen)
                    or any(
                        not url.startswith("https://")
                        or not _host_allowed(url, domains)
                        or not 1 <= len(page.focus.strip()) <= 320
                        for url, page in zip(urls, pages, strict=True)
                    )
                ):
                    raise ValueError(
                        "Choose unread candidates or HTTPS pages within official domains."
                    )
                for url, page in zip(urls, pages, strict=True):
                    vid = self.corpus.register(url)
                    if vid in ids or vid in seen:
                        raise ValueError("The approved page was already selected or read.")
                    direct[vid] = {"role": page.role, "focus": page.focus}
                    self.corpus.candidates[vid]["discovery_query"] = page.focus
                    ids.append(vid)
                self.corpus.persist()
            except ValueError as exc:
                return json.dumps({"error": str(exc)})
            outputs = []
            if followup:
                freshness["article_attempted"] = True
            for vid in ids:
                seen.append(vid)
                self.save()
                receipt = json.loads(
                    await self.reader.ainvoke({"version_id": vid, "passage_limit": 6}, config)
                )
                if receipt.get("passages"):
                    self.data["read_ids"].append(vid)
                    self.observe_release(vid, receipt)
                    if route := direct.get(vid):
                        leads = []
                        for link in discover_links(
                            self.evidence, self.budget, vid, route["focus"], limit=6
                        ):
                            if not _host_allowed(link["url"], domains) or not link[
                                "url"
                            ].startswith("https://"):
                                continue
                            target = self.corpus.register(link["url"], link["anchor"])
                            direct.setdefault(target, route)
                            self.corpus.candidates[target]["discovery_query"] = route["focus"]
                            leads.append(
                                {"version_id": target, "url": link["url"], "title": link["anchor"]}
                            )
                        self.corpus.persist()
                        receipt["official_page_links"] = leads
                outputs.append(receipt)
                self.save()
            return json.dumps({"originals": outputs}, ensure_ascii=False)

        @tool
        async def record_current_result(
            outcome: Literal["confirmed", "not_found", "insufficient"],
            text: str,
            passage_ids: list[str],
            actual_units_quote: str | None = None,
            record_on: date | None = None,
            data_through: date | None = None,
        ) -> str:
            """Close with a concise public answer and selected original passage IDs.

            For sales, confirmed also requires actual_units_quote: the exact first-party sentence identifying the product and its actual units sold or shipped; distinguish these metrics.
            confirmed requires the exact requested fact and the latest applicable original
            date (record_on). If the sources lack the requested disclosure, use not_found only
            after reading two relevant first-party originals, including one from requested_fact.
            Select the supplied passage IDs directly; the system preserves the saved text.
            If the original gives the observation date or reporting period end of actual
            sales, supply data_through separately. A publication date alone does not
            establish data_through. In the previous-period step, retain the newest
            verified comparable value, even if the present value is unknown.
            State the scope of that check and any confirmed context; never infer zero or universal
            non-disclosure. Otherwise use insufficient. A cutoff is not a disclosure date.
            """
            previous = self.data.get("previous_period", {})
            assessing_previous = previous.get("stage") == "assess"
            try:
                if (
                    (self.data.get("result") and not assessing_previous)
                    or not text.strip()
                    or len(text) > 650
                    or len(passage_ids) > 8
                ):
                    raise ValueError("Provide a single concise current answer.")
                passages = [self.evidence.passages[pid] for pid in dict.fromkeys(passage_ids)]
                refs = self.refs_for(passage_ids)
                if assessing_previous:
                    old_refs = (self.data.get("result") or {}).get("references", [])
                    refs = list({r["passage_id"]: r for r in [*refs, *old_refs]}.values())[:12]
                versions = {ref["version_id"] for ref in refs}
                if versions - set(self.data["read_ids"]):
                    raise ValueError("Cite only originals read by this latest check.")
                fact_ids = {
                    vid
                    for a in self.data["attempts"]
                    for vid in a["hits"].get("requested_fact", [])
                }
                fact_ids.update(previous.get("official_ids", []))
                update_ids = {
                    vid for a in self.data["attempts"] for vid in a["hits"].get("latest_update", [])
                }
                for vid, route in self.data.get("official_pages", {}).items():
                    (fact_ids if route["role"] == "requested_fact" else update_ids).add(vid)
                freshness = self.data.get("freshness", {})
                release = freshness.get("release")
                if release:
                    update_ids.update(ref["version_id"] for ref in release["references"])
                record = record_on.isoformat() if record_on else None
                through = data_through.isoformat() if data_through else None
                if record and record_on > self.cutoff:
                    raise ValueError("The source record date exceeds the investigation cutoff.")
                if record and not any(record == day for p in passages for day, *_ in dates(p.text)):
                    raise ValueError("The record date must occur in the cited original quotes.")
                if outcome == "confirmed" and not (
                    record
                    and versions & fact_ids
                    and (not assessing_previous or all(p.version_id in fact_ids for p in passages))
                    and any(
                        p.version_id in fact_ids and record in {day for day, *_ in dates(p.text)}
                        for p in passages
                    )
                ):
                    raise ValueError(
                        "A confirmed result needs the recent dated record of the asked fact."
                    )
                if self.sales_question and outcome == "confirmed":
                    quote = actual_units_quote or ""
                    if not (
                        QUANTITY.search(quote)
                        and re.search(
                            r"sold|shipped|sell.through|sell.in|售出|销量|出货|販売本数",
                            quote,
                            re.I,
                        )
                        and not re.search(
                            r"forecast|projected|expected|predict|estimate|预期|估计|预估",
                            quote,
                            re.I,
                        )
                        and not PREORDERS.search(quote)
                        and any(p.version_id in fact_ids and quote in p.text for p in passages)
                    ):
                        raise ValueError(
                            "Actual product units need a matching first-party original sentence with the number and actual metric."
                        )
                if outcome == "not_found" and (
                    len(versions) < 2 or not versions & fact_ids or not versions & update_ids
                ):
                    raise ValueError("Read two relevant original pages covering both search lines.")
                if data_through and not (
                    outcome == "confirmed"
                    and record_on
                    and data_through <= record_on
                    and any(
                        p.version_id in fact_ids and through in {day for day, *_ in dates(p.text)}
                        for p in passages
                    )
                ):
                    raise ValueError(
                        "The reporting cutoff must occur in a cited record of actual sales."
                    )
                if freshness.get("stale_sources") and not release and outcome != "confirmed":
                    outcome, text, record = "insufficient", UNKNOWN_LAUNCH_ANSWER, None
                elif UNSUPPORTED_PRELAUNCH.search(text):
                    outcome = "not_found" if release and outcome == "not_found" else "insufficient"
                    text = "所查官方记录未提供可核实的实际销量。"
                    record = None
                if assessing_previous:
                    previous["stage"] = "done"
                self.data["result"] = {
                    "outcome": outcome,
                    "text": text.strip(),
                    "record_on": record,
                    **({"data_through": through} if through and outcome == "confirmed" else {}),
                    "references": refs,
                }
                self.save()
                if outcome == "confirmed" or not self.begin_previous_period():
                    self.budget.finish_research("current_fact_checked")
                return json.dumps(result_view(self.data), ensure_ascii=False)
            except (ValueError, KeyError) as exc:
                return json.dumps({"error": str(exc)}, ensure_ascii=False)

        return [
            search_current_sources,
            recover_official_sources,
            read_current_sources,
            refresh_current_status,
            search_previous_sales,
            record_current_result,
        ]

    def report(self, draft=None):
        view = written_view(self.data, draft)
        confirmed = view["outcome"] == "confirmed"
        refs = " ".join(f"[{ref['passage_id']}]" for ref in view.get("references", []))
        if view.get("lead_ids"):
            refs = " ".join(f"[{pid}]" for pid in view["lead_ids"])
        findings = []
        for point in view.get("other_findings", []):
            attribution = "官方披露" if point["official"] else "第三方估算"
            citations = " ".join(f"[{pid}]" for pid in point["citations"])
            findings.extend(
                [f"{attribution}（{point['source']}） · {point['summary']} {citations}", ""]
            )
        if notes := report_preorder_notes(self.data, draft):
            findings.extend(["## 已查到的预购信息", ""])
            findings.extend(
                f"{note['label']} · {note.get('summary', note['text'])} [{note['passage_id']}]\n"
                for note in notes
            )
        if release := view.get("release"):
            launch_refs = " ".join(f"[{ref['passage_id']}]" for ref in release["references"])
            findings.extend([f"官方 {release['day']} 公告确认已发售。{launch_refs}", ""])
        heading = view["label"] if confirmed else "核查结论"
        conclusion = [f"## {heading}", "", f"核查截至 {view['cutoff_label']}。", view["text"], refs]
        if view.get("lead_source"):
            conclusion.append(
                "依据 "
                + view["lead_source"]
                + (" · 公告日期 " + view["lead_published"] if view.get("lead_published") else "")
            )
        if view["record_on"]:
            conclusion.extend(["", f"原始记录日期 {view['record_on']}"])
        if cutoff := view.get("data_through"):
            conclusion.append(f"数据截至 {cutoff}")
        if confirmed and view["release_pending"]:
            conclusion.append("本轮未取得更新的发售状态公告。")
        conclusion.append("")
        lines = ["# 最新消息核查", ""]
        lines.extend(conclusion + findings)
        return "\n".join(lines).strip() + "\n"
