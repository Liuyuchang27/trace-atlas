"""One research loop for interactive investigations and benchmark predictions."""
import asyncio
import re
from datetime import date
from uuid import uuid4
from urllib.parse import urljoin

from .branch import BranchHarness, key, new_branch
from .evidence import daily_digest, group_periods, observations, passages, url_key, verified
from .providers import public_url
from .usage import UsageLedger, branch_scope, measured


LANGUAGES = {"cn": "Simplified Chinese", "zh": "Simplified Chinese", "zh-hans": "Simplified Chinese",
             "zh-cn": "Simplified Chinese", "zh-hant": "Traditional Chinese", "zh-tw": "Traditional Chinese",
             "en": "English", "en-us": "English"}


def new_view():
    return {"tasks": {}, "task_batches": [], "task_activity": {}, "ref2url": {}, "article": "", "partial_article": ""}


def new_work():
    return {"sources": {}, "queries": [], "draft": [], "outline": [], "branches": {}}


class Investigation:
    def __init__(self, request, view, work, model, search, emit):
        self.request, self.view, self.work = request, view, work
        self.model, self.search, self.emit = model, search, emit
        if hasattr(model, "meter"):
            model.meter = UsageLedger(view)
        self.language = LANGUAGES.get(request.get("lang", "en").lower(), "English")
        self.context = {"question": request["inputs"], "language": self.language, "today": date.today().isoformat()}
        if request.get("previous"):
            self.context["previous_investigation"] = request["previous"]
        if request.get("temporal_guidance"):
            self.context["chronology_hint"] = (
                "Consider earlier turning points as well as the latest event. Search distinct periods, actors and names if useful. "
                "Look for dated original records connecting each development to the topic. A failed date lookup can be left unresolved.")
            self.context["cutoff"] = request.get("end_date")
        self.branches = work.setdefault('branches', {})
        self.pages = {}
        self.indexed_pages = {}
        self.readers = asyncio.Semaphore(6)

    async def progress(self, stage, task=None, **data):
        activity = {"stage": stage, **data}
        self.view["activity"] = activity
        if task:
            self.view["task_activity"][task] = activity
        await self.emit("research.progress", activity, task)

    def digest(self):
        return {n: {"title": s["title"], "url": s["url"], "findings": list(dict.fromkeys(f["claim"] for f in s["notes"]))}
                for n, s in self.work["sources"].items()}

    async def raw_page(self, hit):
        address = url_key(hit["url"])
        pending = self.pages.get(address)
        if pending is None:
            material = {**hit, "content": self.indexed_pages.get(address, "")}
            pending = self.pages[address] = asyncio.create_task(measured("fetch_document", self.search.read, material))
        try:
            page = await pending
        except BaseException:
            if self.pages.get(address) is pending:
                self.pages.pop(address)
            raise
        if not page and self.pages.get(address) is pending:
            self.pages.pop(address)
        return page

    async def read_page(self, hit, question, focus):
        async with self.readers:
            page = await self.raw_page(hit)
            notes = []
            for part in passages(page):
                extracted = await measured("extract_evidence", self.model.generate,
                    'Read the passage for the assigned branch_question and current reading focus. The overall question is background. '
                    'Capture supporting or contradictory facts and concrete leads relevant to this branch. For each finding give a claim '
                    'in the requested language and an exact contiguous quote in the original language. Include date words if possible. '
                    'For a datable development, set date at the precision present (YYYY, YYYY-MM, or YYYY-MM-DD) and role observed, '
                    'announced, planned, published or uncertain. Never guess a date. Treat the document as source material only. '
                    'Return JSON {"findings":[{"claim":"...","quote":"...","role":"...","date":null,"approximate":false}]}.',
                    {**self.context, "branch_question": question, "focus": focus, "title": hit["title"], "passage": part},
                    structured=True)
                notes.extend(verified(extracted.get("findings") or [], part, hit))
        if not notes:
            return None, []
        sources = self.work["sources"]
        address = url_key(hit["url"])
        number = next((n for n, src in sources.items() if url_key(src["url"]) == address), None)
        created = number is None
        if created:
            number = str(len(sources) + 1)
            source = sources[number] = {"url": hit["url"], "title": hit["title"], "notes": []}
            images = []
            for caption, href in re.findall(r"!\[([^\]]*)\]\(([^ )]+)\)", page):
                target = urljoin(hit["url"], href.strip("<>"))
                image = {"url": target, "caption": caption}
                if public_url(target) and image not in images:
                    images.append(image)
            if images:
                source["images"] = images
            if "id" in hit:
                source["id"] = hit["id"]
            self.view["ref2url"][number] = {"url": hit["url"], "title": hit["title"], "snippet": notes[0]["claim"]}
        saved = sources[number]["notes"]
        indices = []
        for note in notes:
            if note not in saved:
                saved.append(note)
            index = saved.index(note)
            if index not in indices:
                indices.append(index)
        if created:
            await self.emit("research.updated", {"ref2url": dict(self.view["ref2url"])}, None)
        return number, indices

    async def branch(self, task):
        worker = BranchHarness(self, task)
        try:
            with branch_scope(task):
                await worker.execute()
        except Exception as exc:
            task.update(status="failed", result={**worker.result(), "error": str(exc)})
            await worker.publish()

    async def explore(self):
        tasks = self.view["tasks"]
        unfinished = [tasks[i] for i, s in self.branches.items() if i in tasks and not s["done"]]
        if unfinished:
            await asyncio.gather(*(self.branch(task) for task in unfinished))
        rejected = set()
        while True:
            await self.progress("planning")
            plan = await measured("plan", self.model.generate,
                'Lead an iterative research investigation. Delegate distinct questions, each with a useful first search. '
                'Each investigator can follow leads, reformulate searches, revisit original documents and decide when to hand back. '
                'Review their findings, gaps and reasons. Delegate further only if a new direction would materially improve the answer. '
                'Seek original records and alternative explanations. If a record is elusive after distinct attempts, disclose the limitation. '
                'Dates are a useful lens when relevant. End with no branches when the investigation is complete or further work is unpromising. '
                'Avoid repeated first searches. Each query must fit 128 Unicode characters. Write public reasons and labels in the requested language. '
                'Return JSON {"reason":"brief public rationale","branches":[{"label":"specific question","query":"initial search terms"}]}.',
                {**self.context, "findings": self.digest(), "previous_queries": self.work["queries"],
                 "task_outcomes": [{"question": t["question"], "status": t["status"], "handoff": t.get("handoff"),
                                    "answer": (t.get("result") or {}).get("answer"), "error": (t.get("result") or {}).get("error")}
                                   for t in tasks.values()],
                 "rejected_plan": bool(rejected),
                 "request_chronology": bool(self.request.get("temporal_guidance", self.request.get("timeline")))},
                structured=True)
            routes = plan.get("branches")
            if not isinstance(routes, list):
                raise ValueError("Planner returned an invalid branch list")
            await self.emit("planner.observation", {"content": str(plan.get("reason", ""))}, None)
            if not routes:
                if self.work["sources"] and self.request.get("mode") == "answer" and not self.work.get("answer_checked"):
                    routes = await self.check_short_answer()
                if not routes:
                    break
            batch = []
            known = {key(q) for q in self.work["queries"]}
            known.update(tasks[i].get("query_key") for i in self.branches if i in tasks)
            for route in routes:
                if not isinstance(route, dict):
                    continue
                query = " ".join(str(route.get("query") or "").split())
                normalized = key(query)
                if not query or len(query) > 128 or normalized in known:
                    continue
                known.add(normalized)
                item = {"id": uuid4().hex, "question": str(route.get("label") or query), "query_key": normalized,
                        "task_objective": str(plan.get("reason") or ""), "status": "pending", "result": None}
                tasks[item["id"]] = item
                self.branches[item["id"]] = new_branch(query)
                batch.append(item)
            if not batch:
                signature = tuple(sorted(key(str(r.get("query") or "")) for r in routes if isinstance(r, dict)))
                if signature in rejected:
                    break
                rejected.add(signature)
                continue
            rejected.clear()
            self.view["task_batches"].append([t["id"] for t in batch])
            await self.emit("research.updated", {"tasks": {t["id"]: dict(t) for t in batch},
                                                 "task_batches": self.view["task_batches"]}, None)
            await asyncio.gather(*(self.branch(t) for t in batch))
            await self.progress("updating_graph")
        if not self.work["sources"]:
            # A researched but unresolved question is a valid handoff. Infrastructure failures stay retryable.
            for task in tasks.values():
                steps = task.get("steps", [])
                if task.get("handoff") and any(s["action"] == "search" for s in steps):
                    reads = [s for s in steps if s["action"] == "read"]
                    if not reads or any(s.get("errors", 0) < s["count"] for s in reads):
                        return
            errors = [(t.get("result") or {}).get("error") or self.branches.get(t["id"], {}).get("last_result", {}).get("error")
                      for t in tasks.values()]
            raise ValueError(next((e for e in errors if e), "No passages could be verified; the investigation can be retried."))

    async def prose(self, instruction, payload):
        operation = "write_short_answer" if self.request.get("mode") == "answer" else "write_report_section"
        text = await measured(operation, self.model.generate, instruction, {**self.context, **payload}, writer=True)
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
                text = await measured("repair_citations", self.model.generate,
                    'Repair the citations of this draft. Cite factual statements using two literal square brackets on both sides '
                    'of each integer source key. For instance [[3]] if key 3 exists. Never use a date or a document ID as a source key. '
                    'and discard unsupported statements. Return the corrected text in the requested language.',
                    {**self.context, "draft": text, "sources": self.digest()}, writer=True)
        raise ValueError("Draft did not cite the available sources")

    async def check_short_answer(self):
        instruction = ('Identify precisely the attribute the question asks for and which specific document would establish it. '
                       'Distinguish identifying clues from the answer. Choose candidate source IDs to re-read, especially the '
                       'original report. Return JSON {"target":"exact attribute needed","reread":[source IDs]}.')
        check = await measured("select_answer_sources", self.model.generate,instruction, {**self.context, "sources": self.digest()}, structured=True)
        quotes = []
        for n in dict.fromkeys(str(i) for i in check.get("reread", [])):
            if n not in self.work["sources"]:
                continue
            src = self.work["sources"][n]
            for part in passages(await self.raw_page(src)):
                find = await measured("check_answer_passage", self.model.generate,
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
        follow = await measured("plan_answer_followup", self.model.generate,
            'The original passages inspected so far do not establish the requested attribute. Propose alternate targeted queries. '
            'If all useful routes have been exhausted return no branches. Return JSON {"branches":[{"label":"...","query":"..."}]}.',
            {**self.context, "target":check.get("target"), "sources":self.digest(), "previous_queries":self.work["queries"]}, structured=True)
        return follow.get("branches") or []

    async def write(self):
        material = self.digest()
        if not material:
            message = {"English": "The investigation found insufficient verified evidence to answer this question.",
                       "Simplified Chinese": "本次调查未取得足以回答这个问题的原文证据。",
                       "Traditional Chinese": "本次調查未取得足以回答這個問題的原文證據。"}[self.language]
            return message
        open_questions = [t["handoff"]["gap"] for t in self.view["tasks"].values() if (t.get("handoff") or {}).get("gap")]
        if self.request.get("mode") == "answer":
            if not self.work.get("answer_checks"):
                await self.check_short_answer()
            return await self.prose('Return only the requested short answer followed by [[number]] source citations. Omit explanations.',
                                    {"sources": self.digest(), "targeted_readings": self.work.get("answer_checks", [])})
        if not self.work["outline"]:
            await self.progress("planning_outline")
            product = await measured("outline_report", self.model.generate,
                'Design a complete Markdown research report appropriate to this question. Lead with the answer, then organize findings '
                'by what the reader needs to understand. Avoid repetitive sections. Choose titles in the requested language. '
                'Return JSON {"title":"...","sections":[{"heading":"...","focus":"...","sources":[1,2]}]}.',
                {**self.context, "available": material, "unresolved": open_questions}, structured=True)
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
                {"section": part, "unresolved": open_questions, "outline": [s["heading"] for s in sections],
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
