"""Run a delegated research question with its own observations and tool decisions."""
import asyncio
import re
import unicodedata

import httpx

from .evidence import url_key
from .usage import measured


INSTRUCTIONS = """You are responsible for the assigned branch_question within the overall investigation.
Choose your next action from the current evidence and tool observations. You may search again, read more of the
results already found, revisit a document with a more precise focus, or finish. Identify which exact attribute,
event or relationship the question asks about. Use new names and leads from original passages to change direction.
Search snippets and peer leads are leads. When a candidate plausibly addresses a gap, inspect the original before
spending more searches on variations. Peers may share a name or a document while you work. Follow it if useful;
only your own numbered verified findings may support your handoff. Match each citation to its URL in source_index.
When a named authority is central, consider a domain-specific web search. If the available passage is a third party's
account, attribute the finding to that source and qualify your conclusion accordingly. Give each useful lead the follow-up it warrants. Once the assigned question is answered, finish even if other branches have work to do. If further work is unlikely
to help, hand back what is supported and state what remains unresolved. Decide for yourself when the evidence suffices.
Return one JSON object. The action forms are:
{"action":"search","query":"a new search query, up to 128 characters","reason":"what this could resolve"}
{"action":"read","pages":[1,2],"focus":"what to look for in these documents","reason":"why examine them"}
{"action":"finish","answer":"your supported conclusion with [[source number]] citations","gap":"remaining uncertainty, or empty","reason":"why stop here"}
Use candidate page numbers for read and verified source numbers for citations. Write public reasons, findings and gaps
in the requested language. A reason is one short operational explanation."""


def key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def new_branch(query):
    return {"pending": {"action": "search", "query": query}, "catalog": {}, "notes": {},
            "queries": [], "read_at": {}, "last_result": {}, "blocked": [], "done": False}


class BranchHarness:
    def __init__(self, run, task):
        self.run = run
        self.task = task
        self.state = run.work["branches"][task["id"]]
        self.catalog = self.state["catalog"]
        self.steps = task.setdefault("steps", [])

    async def publish(self):
        await self.run.emit("research.updated", {"task": self.task}, self.task["id"])

    def findings(self):
        return {n: [self.run.work["sources"][n]["notes"][i] for i in indices]
                for n, indices in self.state["notes"].items()}

    def result(self, answer=""):
        findings = self.findings()
        sources = {n: {**self.run.view["ref2url"][n], "snippet": notes[0]["claim"]} for n, notes in findings.items()}
        if not answer:
            answer = "\n\n".join(f'{note["claim"]} [[{n}]]' for n, notes in findings.items() for note in notes)
        return {"answer": answer, "references": sources, "evidence_records": [
            {**sources[n], "evidence_quote": note["quote"]} for n, notes in findings.items() for note in notes]}

    def context(self):
        # A sibling's first verified passage can supply a name or a useful URL before its handoff.
        by_url = {url_key(hit["url"]): number for number, hit in self.catalog.items()}
        peer_leads = {}
        for source_id, source in self.run.work["sources"].items():
            if source_id in self.state["notes"] or not source["notes"]:
                continue
            address = url_key(source["url"])
            number = by_url.get(address)
            if number is None:
                number = str(len(self.catalog) + 1)
                by_url[address] = number
                self.catalog[number] = {k: source[k] for k in ("title", "url", "id") if k in source}
                self.catalog[number]["source_date"] = source["notes"][0].get("source_date")
                self.state["blocked"].clear()
            peer_leads[number] = source["notes"][0]["claim"]
        recent = set(self.state["last_result"].get("pages", []))
        candidates = [{"page": int(i), "title": hit["title"], "url": hit["url"],
                       **({"snippet": hit.get("snippet", "")} if i in recent else {}),
                       **({"peer_lead": peer_leads[i]} if i in peer_leads else {})}
                      for i, hit in self.catalog.items()]
        return {**self.run.context, "branch_question": self.task["question"], "history": self.steps,
                "candidates": candidates, "verified_sources": self.findings(),
                "source_index": {n: {k: self.run.work["sources"][n][k] for k in ("title", "url")}
                                 for n in self.state["notes"]},
                "last_observation": self.state["last_result"]}

    async def search(self, action):
        query = str(action.get("query") or "").strip()
        if not query or len(query) > 128:
            raise ValueError("Provide a search query of 1–128 characters.")
        normalized = key(query)
        if normalized in self.state["queries"]:
            raise ValueError("This branch already ran this query. Read a result, change the terms, or finish.")
        await self.run.progress("searching", self.task["id"], query=query)
        hits = await measured("search", self.run.search.search, query)
        by_url = {url_key(h["url"]): i for i, h in self.catalog.items()}
        fresh = False
        pages = []
        for hit in hits:
            url = url_key(hit["url"])
            number = by_url.get(url)
            if number is None:
                number = str(len(self.catalog) + 1)
                by_url[url] = number
                self.catalog[number] = {k: v for k, v in hit.items() if k != "content"}
                fresh = True
            if hit.get("content"):
                self.run.indexed_pages[url] = hit["content"]
            if number not in pages:
                pages.append(number)
        self.state["queries"].append(normalized)
        if normalized not in self.run.work["queries"]:
            self.run.work["queries"].append(normalized)
        self.steps.append({"action": "search", "query": query, "reason": str(action.get("reason") or ""), "count": len(pages)})
        return {"pages": pages, "new_candidates": fresh}, fresh

    async def read(self, action):
        pages = action.get("pages")
        if not isinstance(pages, list) or not pages or any(type(p) is not int or str(p) not in self.catalog for p in pages):
            raise ValueError("Read one or more numbered pages from this branch's search results.")
        focus = str(action.get("focus") or self.task["question"]).strip()
        focus_key = key(focus)
        selected = [str(i) for i in dict.fromkeys(pages) if focus_key not in self.state["read_at"].get(str(i), [])]
        if not selected:
            raise ValueError("These pages were already read with this focus. Choose a different lead or finish.")
        await self.run.progress("reading", self.task["id"], title=focus)
        outcomes = await asyncio.gather(*(self.run.read_page(self.catalog[i], self.task["question"], focus)
                                          for i in selected), return_exceptions=True)
        errors = []
        gained = 0
        for i, outcome in zip(selected, outcomes):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, httpx.HTTPStatusError) and outcome.response.status_code in {401, 403}:
                raise outcome
            if isinstance(outcome, Exception):
                errors.append({"page": int(i), "error": str(outcome)})
                continue
            self.state["read_at"].setdefault(i, []).append(focus_key)
            source, indices = outcome
            if source:
                known = self.state["notes"].setdefault(source, [])
                extra = [index for index in indices if index not in known]
                known.extend(extra)
                gained += len(extra)
        self.steps.append({"action": "read", "focus": focus, "reason": str(action.get("reason") or ""),
                           "count": len(selected), "new_findings": gained, "errors": len(errors)})
        observation = {"pages": selected, "new_findings": gained, "errors": errors}
        return observation, bool(gained)

    async def execute(self):
        self.task["status"] = "running"
        await self.publish()
        while not self.state["done"]:
            action = self.state["pending"]
            if action is None:
                await self.run.progress("branch_review", self.task["id"], title=self.task["question"])
                action = await measured("branch_decision", self.run.model.generate, INSTRUCTIONS, self.context(), structured=True)
                self.state["pending"] = action
                await self.publish()
            tool = action.get("action")
            try:
                if tool == "finish":
                    answer = str(action.get("answer") or "").strip()
                    cited = set(re.findall(r"\[\[(\d+)\]\]", answer))
                    grounded = bool(self.state["notes"])
                    if cited - self.state["notes"].keys() or (grounded and answer and not cited):
                        raise ValueError("Use only this branch's verified source numbers, or omit the answer.")
                    self.task["result"] = self.result(answer if grounded else "")
                    self.task["handoff"] = {"kind": "agent", "reason": str(action.get("reason") or ""),
                                            "gap": str(action.get("gap") or ("" if grounded else self.task["question"]))}
                    break
                if tool == "search":
                    observation, advanced = await self.search(action)
                elif tool == "read":
                    observation, advanced = await self.read(action)
                else:
                    raise ValueError("Choose search, read or finish.")
            except (ValueError, httpx.HTTPError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {401, 403}:
                    raise
                observation, advanced = {"error": str(exc)}, False

            # Detect repeated ineffective actions at an unchanged evidence state. Distinct
            # searches with no hits, or a new reading focus, remain valid investigative choices.
            signature = [tool, key(str(action.get("query") or action.get("focus") or "")),
                         sorted(str(i) for i in action["pages"]) if isinstance(action.get("pages"), list) else [],
                         observation.get("error", ""), observation.get("pages", [])]
            if advanced:
                self.state["blocked"].clear()
            elif signature in self.state["blocked"]:
                self.task["handoff"] = {"kind": "loop_guard", "reason": "", "gap": self.task["question"]}
                break
            else:
                self.state["blocked"].append(signature)
            self.state["last_result"] = observation
            self.state["pending"] = None
            self.task["result"] = self.result()
            await self.publish()
        self.state.update(done=True, pending=None)
        self.task["status"] = "completed"
        if not self.task.get("result"):
            self.task["result"] = self.result()
        await self.publish()
