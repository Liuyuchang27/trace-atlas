"""Run one or more input-only benchmark cases through the workbench research loop."""
import argparse
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote

from dotenv import load_dotenv
import httpx

from .providers import Services
from .research import Investigation, new_view, new_work


def save(path, value):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


class Corpus:
    def __init__(self, process):
        self.process = process
        self.lock = asyncio.Lock()
        self.returned, self.opened = set(), set()
        self.counts = {"search": 0, "read": 0}

    async def rpc(self, body):
        async with self.lock:
            self.process.stdin.write((json.dumps(body) + "\n").encode())
            await self.process.stdin.drain()
            line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError("Corpus reader stopped. Check index.log and Java 21.")
            return json.loads(line)

    async def search(self, query):
        self.counts["search"] += 1
        hits = await self.rpc({"search": query})
        self.returned.update(h["id"] for h in hits)
        return [{**h, "url": "https://browsecomp-plus.invalid/doc/" + h["id"]} for h in hits]

    async def read(self, hit):
        identifier = hit["id"]
        if identifier not in self.returned:
            raise ValueError("Select a retrieved corpus document")
        result = await self.rpc({"open": identifier})
        self.opened.add(identifier)
        self.counts["read"] += 1
        return result


@asynccontextmanager
async def retrieval(index, services, folder):
    if not index:
        yield services
        return
    jar = index / "anserini-1.2.0-fatjar.jar"
    if not jar.is_file() or not any((index/"bm25").glob("segments_*")):
        raise FileNotFoundError("Expected the official BM25 index and Anserini 1.2.0 jar")
    with (folder / "index.log").open("w") as errors:
        proc = await asyncio.create_subprocess_exec("java", "-cp", str(jar), str(Path(__file__).with_name("Corpus.java")),
            str(index/"bm25"), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=errors, limit=128*1024*1024)
        try:
            yield Corpus(proc)
        finally:
            proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()


def submission(suite, case, view, tools):
    article = view["article"]
    refs = view["ref2url"]
    if suite == "deepresearch-bench":
        article = re.sub(r"\[\[(\d+)\]\]", lambda m: "[" + m[1] + "](" + quote(refs[m[1]]["url"], safe=":/%?&=+#") + ")", article)
        return {"id": case["id"], "prompt": case.get("prompt") or case["question"], "article": article}
    if suite == "browsecomp-plus":
        return {"query_id": str(case["id"]), "retrieved_docids": sorted(tools.returned), "opened_docids": sorted(tools.opened),
                "tool_call_counts": tools.counts, "result": [{"type":"output_text", "output": re.sub(r"\s*\[\[\d+\]\]", "", article).strip()}]}
    if "daily_timeline" in view:
        return {"predict-timeline":view["daily_timeline"]}
    days = defaultdict(list)
    for row in view.get("timeline", []):
        day = row["date"]
        if (len(day) == 10 and day <= (case.get("end_date") or "9999-12-31")
                and row["date_basis"] == "event" and row["role"] in {"observed","announced"}
                and row["event"] not in days[day]):
            days[day].append(row["event"])
    return {"predict-timeline": [{"start":d, "events":days[d]} for d in sorted(days)]}


async def run(args, rows):
    failed = False
    root = Path(__file__).parent
    fingerprint = hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in sorted(root.iterdir()) if p.suffix in {".java",".py"})).hexdigest()
    async with httpx.AsyncClient(timeout=180) as client:
        for case in rows:
            folder = args.out / str(case["id"])
            folder.mkdir(parents=True, exist_ok=False)
            svc = Services(client, fallback=False)
            view, work = new_view(), new_work()
            request = {"inputs":case.get("prompt") or case["question"], "lang":case.get("language") or "en",
                       "mode":{"browsecomp-plus":"answer", "open-tls":"timeline", "deepresearch-bench":"report"}[args.suite],
                       "timeline":args.suite == "open-tls", "temporal_guidance":args.temporal_guidance, "end_date":case.get("end_date")}
            meta = {"id":case["id"], "suite":args.suite, "status":"running", "kernel_sha256":fingerprint,
                    "manifest_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(), "temporal_guidance":args.temporal_guidance,
                    "retrieval":"fixed-bm25" if args.index else "deepseek", "results_per_search":12, "models":[svc.flash,svc.writer]}
            if args.replay_evidence:
                saved = args.replay_evidence.read_bytes()
                work["sources"] = json.loads(saved)
                meta["evidence_replay_sha256"] = hashlib.sha256(saved).hexdigest()
                meta["retrieval"] = "saved-evidence"
            save(folder / "run.json", meta)
            with (folder / "events.jsonl").open("w", encoding="utf-8") as events:
                async def emit(kind, data, task):
                    events.write(json.dumps(dict(type=kind, schema_version=2, task_id=task, data=data), ensure_ascii=False)+"\n")
                    events.flush()
                try:
                    async with retrieval(args.index, svc, folder) as search:
                        await Investigation(request, view, work, svc, search, emit).run(write_only=bool(args.replay_evidence))
                        pred = submission(args.suite, case, view, search)
                        save(folder/"prediction.json", pred)
                        if args.suite == "deepresearch-bench":
                            (folder/"submission.jsonl").write_text(json.dumps(pred,ensure_ascii=False)+"\n",encoding="utf-8")
                        meta["status"] = "completed"
                except Exception as error:
                    meta.update(status="failed", error=str(error))
                    failed = True
                finally:
                    (folder/"report.md").write_text(view["article"] or view["partial_article"], encoding="utf-8")
                    save(folder/"evidence.json", work["sources"])
                    if request["timeline"]:
                        save(folder/"timeline.json", view.get("timeline", []))
                    meta["usage"] = dict(svc.usage)
                    save(folder/"run.json", meta)
                    print(f"{args.suite}/{case['id']} {meta['status']}", flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["open-tls","deepresearch-bench","browsecomp-plus"])
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--id", action="append")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--temporal-guidance", action="store_true")
    parser.add_argument("--replay-evidence", type=Path, help="Rebuild one Open-TLS prediction from the same previously read sources")
    args = parser.parse_args()
    if (args.suite == "browsecomp-plus") != bool(args.index):
        parser.error("--index is required exclusively for BrowseComp-Plus")
    if args.temporal_guidance and args.suite != "open-tls":
        parser.error("Temporal planning ablations apply to Open-TLS")
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("Environment file does not exist")
        load_dotenv(args.env_file)
    raw = args.queries.read_text(encoding="utf-8")
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()] if args.queries.suffix == ".jsonl" else json.loads(raw)
    if isinstance(rows, dict):
        rows = rows["items"]
    chosen = [r for r in rows if not args.id or str(r["id"]) in args.id]
    if not chosen or (args.id and set(args.id) != {str(r["id"]) for r in chosen}):
        parser.error("Unknown or empty case selection")
    if args.replay_evidence:
        if len(chosen) != 1 or args.suite != "open-tls" or args.replay_evidence.name != "evidence.json":
            parser.error("Evidence replay requires one Open-TLS case and its evidence.json")
        original = json.loads(args.replay_evidence.with_name("run.json").read_text())
        if original["suite"] != args.suite or str(original["id"]) != str(chosen[0]["id"]):
            parser.error("Evidence belongs to a different benchmark case")
    for r in chosen:
        if not re.fullmatch(r"[\w-][\w.-]*", str(r["id"])) or not (r.get("prompt") or r.get("question")):
            parser.error("Invalid case identifier or missing question")
    raise SystemExit(int(asyncio.run(run(args, chosen))))


if __name__ == "__main__":
    main()
