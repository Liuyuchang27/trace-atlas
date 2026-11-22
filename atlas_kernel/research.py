"""One research loop for interactive investigations and benchmark predictions."""
import asyncio
import re
from datetime import date
from uuid import uuid4
from urllib.parse import urljoin

from .evidence import daily_digest, group_periods, observations, passages, url_key, verified
from .providers import public_url


LANGUAGES = {"cn": "Simplified Chinese", "zh": "Simplified Chinese", "zh-hans": "Simplified Chinese",
             "zh-cn": "Simplified Chinese", "zh-hant": "Traditional Chinese", "zh-tw": "Traditional Chinese",
             "en": "English", "en-us": "English"}


def new_view():
    return {"tasks": {}, "task_batches": [], "task_activity": {}, "ref2url": {}, "article": "", "partial_article": ""}


def new_work():
    return {"sources": {}, "queries": [], "draft": [], "outline": [], "stagnation": 0}


class Investigation:
    def __init__(self, request, view, work, model, search, emit):
        self.request, self.view, self.work = request, view, work
        self.model, self.search, self.emit = model, search, emit
        self.language = LANGUAGES.get(request.get("lang", "en").lower(), "English")
        self.context = {"question": request["inputs"], "language": self.language, "today": date.today().isoformat()}
        if request.get("previous"):
            self.context["previous_investigation"] = request["previous"]
        if request.get("temporal_guidance"):
            self.context["chronology_hint"] = (
                "Consider earlier turning points as well as the latest event. Search distinct periods, actors and names if useful. "
                "Look for dated original records connecting each development to the topic. A failed date lookup can be left unresolved.")
            self.context["cutoff"] = request.get("end_date")
        self.pages = {}
        self.readers = asyncio.Semaphore(6)
        self.branch_slots = asyncio.Semaphore(4)

    async def progress(self, stage, task=None, **data):
        activity = {"stage": stage, **data}
        self.view["activity"] = activity
        if task:
            self.view["task_activity"][task] = activity
        await self.emit("research.progress", activity, task)

    def digest(self):
        return {n: {"title": s["title"], "url": s["url"], "findings": list(dict.fromkeys(f["claim"] for f in s["notes"]))}
                for n, s in self.work["sources"].items()}

    async def read_page(self, hit):
        async with self.readers:
            page = await self.search.read(hit)
            notes = []
            for part in passages(page):
                extracted = await self.model.generate(
                    'Read the passage for the research question. Preserve facts, conflicting findings, numbers and relevant dates. '
                    'For each useful finding give a succinct claim in the requested language and an exact contiguous quote in the original language. '
                    'Include surrounding date words in the quote when possible. For a datable development, set date at the precision present '
                    '(YYYY, YYYY-MM, or YYYY-MM-DD) and role observed, announced, planned, published or uncertain. Never guess a date. '
                    'Return JSON {"findings":[{"claim":"...","quote":"...","role":"...","date":null,"approximate":false}]}.',
                    {**self.context, "title": hit["title"], "passage": part}, structured=True)
                notes.extend(verified(extracted.get("findings") or [], part, hit))
            if not notes:
                return None
            sources = self.work["sources"]
            number = next((n for n, s in sources.items() if url_key(s["url"]) == url_key(hit["url"])), None)
            if number is None:
                number = str(len(sources) + 1)
                sources[number] = {"url": hit["url"], "title": hit["title"], "notes": notes}
                images = []
                for caption, href in re.findall(r"!\[([^\]]*)\]\(([^ )]+)\)", page):
                    target = urljoin(hit["url"], href.strip("<>"))
                    if public_url(target) and {"url":target,"caption":caption} not in images:
                        images.append({"url":target,"caption":caption})
                if images:
                    sources[number]["images"] = images
                if "id" in hit:
                    sources[number]["id"] = hit["id"]
                self.view["ref2url"][number] = {"url": hit["url"], "title": hit["title"], "snippet": notes[0]["claim"]}
                await self.emit("research.updated", {"ref2url": dict(self.view["ref2url"])}, None)
            return number

    async def shared_read(self, hit):
        key = url_key(hit["url"])
        for number, saved in self.work["sources"].items():
            if url_key(saved["url"]) == key:
                return number
        if key not in self.pages:
            self.pages[key] = asyncio.create_task(self.read_page(hit))
        try:
            return await self.pages[key]
        except Exception:
            self.pages.pop(key, None)
            raise

    async def branch(self, task, query):
        async with self.branch_slots:
            task["status"] = "running"
            await self.emit("research.updated", {"task": dict(task)}, task["id"])
            try:
                await self.progress("searching", task["id"], query=query)
                hits = await self.search.search(query)
                ids = []
                if hits:
                    await self.progress("filtering", task["id"], title=task["question"])
                    choice = await self.model.generate(
                        'Select pages to open for this investigation. Prefer primary evidence and complementary independent sources. '
                        'Use one-based entry numbers and select all pages worth reading. Return JSON {"open":[1,2]}.',
                        {**self.context, "investigate": task["question"], "results": [
                            [i, h["title"], h["url"], h["snippet"]] for i, h in enumerate(hits, 1)]}, structured=True)
                    picks = dict.fromkeys(i for i in choice.get("open", []) if type(i) is int and 1 <= i <= len(hits))
                    await self.progress("reading", task["id"], title=task["question"])
                    results = await asyncio.gather(*(self.shared_read(hits[i-1]) for i in picks), return_exceptions=True)
                    errors = [str(r) for r in results if isinstance(r, Exception)]
                    ids = list(dict.fromkeys(r for r in results if isinstance(r, str)))
                    if errors and not ids:
                        raise ValueError(errors[0])
                records = [(n, f) for n in ids for f in self.work["sources"][n]["notes"]]
                task["result"] = {"answer": "\n\n".join(f'{f["claim"]} [[{n}]]' for n, f in records),
                                  "references": {n: self.view["ref2url"][n] for n in ids},
                                  "evidence_records": [{**self.view["ref2url"][n], "evidence_quote": f["quote"]} for n, f in records]}
                task["status"] = "completed"
            except Exception as exc:
                task["result"] = {"answer": "", "references": {}, "error": str(exc)}
                task["status"] = "failed"
            await self.emit("research.updated", {"task": dict(task)}, task["id"])

    async def explore(self):
        known = set(self.work["queries"])
        while self.work["stagnation"] < 2:
            await self.progress("planning")
            tasks = self.view["tasks"]
            plan = await self.model.generate(
                'Lead an iterative research investigation. Decide whether additional searches could materially improve the answer. '
                'Plan independent searches together. Update the plan in response to evidence or failures. Explore alternative explanations '
                'and seek authoritative original records. For institutional status, try a site: query on the issuing institution. '
                'Inspect the domains of the sources actually read; a media account of an official listing is indirect. '
                'If the original record is unavailable after distinct attempts, proceed while disclosing the limitation. '
                'Dates are a useful lens when relevant. With adequate evidence finish with no branches. '
                'Do not repeat queries. Each search query must fit 128 Unicode characters. Write the purpose and branch labels in the requested language. '
                'Return JSON {"reason":"brief public rationale","branches":[{"label":"human-readable question","query":"single search query"}]}.',
                {**self.context, "findings": self.digest(), "previous_queries": list(self.work["queries"]),
                 "task_outcomes": [[t["question"], t["status"], (t.get("result") or {}).get("error")] for t in tasks.values()],
                 "request_chronology": bool(self.request.get("temporal_guidance", self.request.get("timeline"))),
                 "rounds_without_new_sources": self.work["stagnation"]}, structured=True)
            if not isinstance(plan.get("branches"), list):
                raise ValueError("Planner returned an invalid branch list")
            await self.emit("planner.observation", {"content": str(plan.get("reason", ""))}, None)
            if not plan["branches"] and self.work["sources"]:
                if self.request.get("mode") == "answer" and not self.work.get("answer_checked"):
                    additional = await self.check_short_answer()
                    if additional:
                        plan["branches"] = additional
                if not plan["branches"]:
                    break
            batch = []
            for route in plan["branches"]:
                if not isinstance(route, dict):
                    continue
                query = " ".join(str(route.get("query") or "").split())
                key = query.casefold()
                if not query or len(query) > 128 or key in known:
                    continue
                known.add(key)
                self.work["queries"].append(key)
                item = {"id": uuid4().hex, "question": str(route.get("label") or query), "query_key": key,
                        "task_objective": str(plan.get("reason") or ""), "status": "pending", "result": None}
                tasks[item["id"]] = item
                batch.append((item, query))
            before = {note["quote"] for s in self.work["sources"].values() for note in s["notes"]}
            if batch:
                self.view["task_batches"].append([t["id"] for t, _ in batch])
                await self.emit("research.updated", {"tasks": {t["id"]:dict(t) for t,_ in batch},
                                                     "task_batches": self.view["task_batches"]}, None)
                await asyncio.gather(*(self.branch(t, q) for t, q in batch))
                await self.progress("updating_graph")
            advanced = any(note["quote"] not in before for s in self.work["sources"].values() for note in s["notes"])
            self.work["stagnation"] = 0 if advanced else self.work["stagnation"] + 1
        if not self.work["sources"]:
            errors = [(t.get("result") or {}).get("error") for t in self.view["tasks"].values()]
            raise ValueError(next((e for e in errors if e), "No passages verified. Try a different search direction."))

    async def prose(self, instruction, payload):
        text = await self.model.generate(instruction, {**self.context, **payload}, writer=True)
        images = {i["url"] for source in self.work["sources"].values() for i in source.get("images", [])}
        for attempt in range(2):
            # Accept common numbered-source typography; leave Markdown links and tables intact.
            text = re.sub(r"(?<!\[)\[{1,2}(\d+(?:\s*[,，、]\s*\d+)*)\]{1,2}(?![\[(])",
                          lambda m: "".join("[["+n+"]]" for n in re.findall(r"\d+", m[1])), text)
            cited = set(re.findall(r"\[\[(\d+)\]\]", text))
            unknown = cited - self.work["sources"].keys()
            if cited and not unknown:
                return re.sub(r"!\[([^\]]*)\]\(([^ )]+)\)", lambda m: m[0] if m[2].strip("<>") in images else m[1], text)
            if attempt == 0:
                text = await self.model.generate(
                    'Repair the citations of this draft. Cite factual statements using two literal square brackets on both sides '
                    'of each integer source key. For instance [[3]] if key 3 exists. Never use a date or a document ID as a source key. '
                    'and discard unsupported statements. Return the corrected text in the requested language.',
                    {**self.context, "draft": text, "sources": self.digest()}, writer=True)
        raise ValueError("Draft did not cite the available sources")

    async def check_short_answer(self):
        instruction = ('Identify precisely the attribute the question asks for and which specific document would establish it. '
                       'Distinguish identifying clues from the answer. Choose candidate source IDs to re-read, especially the '
                       'original report. Return JSON {"target":"exact attribute needed","reread":[source IDs]}.')
        check = await self.model.generate(instruction, {**self.context, "sources": self.digest()}, structured=True)
        quotes = []
        for n in dict.fromkeys(str(i) for i in check.get("reread", [])):
            if n not in self.work["sources"]:
                continue
            src = self.work["sources"][n]
            for part in passages(await self.search.read(src)):
                find = await self.model.generate(
                    'Read this document for the exact requested attribute. Inspect nearby sentences for antecedents, job titles, '
                    'identifiers and dates. Return JSON {"answer":"minimal exact value","quote":"verbatim contiguous supporting passage"}. '
                    'Leave both fields empty if this passage only identifies the entity or lacks the requested value.',
                    {**self.context, "target":check.get("target"), "document":part}, structured=True)
                support = verified([{"claim":find.get("answer"), "quote":find.get("quote")}], part, src)
                for note in support:
                    if note not in src["notes"]:
                        src["notes"].append(note)
                    quotes.append({"source":n, "answer":note["claim"], "passage":note["quote"]})
        self.work["answer_checks"] = quotes
        if quotes:
            self.work["answer_checked"] = True
            return []
        follow = await self.model.generate(
            'The original passages inspected so far do not establish the requested attribute. Propose alternate targeted queries. '
            'If all useful routes have been exhausted return no branches. Return JSON {"branches":[{"label":"...","query":"..."}]}.',
            {**self.context, "target":check.get("target"), "sources":self.digest(), "previous_queries":self.work["queries"]}, structured=True)
        return follow.get("branches") or []

    async def write(self):
        material = self.digest()
        if self.request.get("mode") == "answer":
            if not self.work.get("answer_checks"):
                await self.check_short_answer()
            return await self.prose('Return only the requested short answer followed by [[number]] source citations. Omit explanations.',
                                    {"sources": self.digest(), "targeted_readings": self.work.get("answer_checks", [])})
        if not self.work["outline"]:
            await self.progress("planning_outline")
            product = await self.model.generate(
                'Design a complete Markdown research report appropriate to this question. Lead with the answer, then organize findings '
                'by what the reader needs to understand. Avoid repetitive sections. Choose titles in the requested language. '
                'Return JSON {"title":"...","sections":[{"heading":"...","focus":"...","sources":[1,2]}]}.',
                {**self.context, "available": material}, structured=True)
            sections = [s for s in product.get("sections", []) if isinstance(s, dict) and isinstance(s.get("heading"), str)]
            if not sections:
                raise ValueError("Report outline contained no sections")
            self.work.update(title=product.get("title") or self.request["inputs"], outline=sections)
        sections = self.work["outline"]
        draft = self.work["draft"]
        for i in range(len(draft), len(sections)):
            part = sections[i]
            await self.progress("writing_section", title=part["heading"], section_index=i+1, section_total=len(sections))
            ids = [str(n) for n in part.get("sources", []) if str(n) in material] or list(material)
            body = await self.prose(
                'Write the body of this report section in the requested language. Answer directly. Give details, competing interpretations '
                'and limits where supported. Distinguish what an original record says from what another outlet reports about it. '
                'Never imply direct inspection of an institution, paper or proof unless the supplied URL and passage establish it. '
                'Use [[number]] citations beside claims. Tables and fenced chart JSON '
                '(type bar or line, title, unit, labels, values) are welcome for sourced data. Embed original images only with a verified '
                'image URL from the supplied material. Omit a repeated section title. Do not invent sources, quotes, images or measurements.',
                {"section": part, "outline": [s["heading"] for s in sections],
                 "sources": {n: self.work["sources"][n] for n in ids}})
            first, _, rest = body.partition("\n")
            if first.lstrip("# ").strip("* ").casefold() == part["heading"].strip("* ").casefold() and rest.strip():
                body = rest.strip()
            draft.append("## " + part["heading"] + "\n\n" + body)
            self.view["partial_article"] = "# " + self.work["title"] + "\n\n" + "\n\n".join(draft)
            await self.emit("research.partial", {"partial_article": self.view["partial_article"]}, None)
        return self.view["partial_article"]

    async def run(self, write_only=False):
        if not write_only and not self.work["outline"]:
            await self.explore()
        article = "" if self.request.get("mode") == "timeline" else await self.write()
        if self.request.get("timeline"):
            await self.progress("building_timeline")
            rows = observations(self.work["sources"])
            self.view["timeline"] = rows
            await self.progress("grouping_timeline", events=len(rows))
            if self.request.get("mode") == "timeline":
                self.view["daily_timeline"] = await daily_digest(self.model, self.request, rows)
            else:
                try:
                    self.view["timeline_view"] = await group_periods(self.model, rows, self.language, self.request["inputs"])
                    self.view["timeline_error"] = False
                except Exception:
                    self.view["timeline_error"] = True
        await self.progress("finalizing")
        self.view.update(article=article, usage=dict(self.model.usage))
        return self.view
