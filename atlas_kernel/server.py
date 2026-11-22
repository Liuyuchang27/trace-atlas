"""Loopback workbench API. A disconnected browser can reattach through the saved run."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, Field
import uvicorn

from .evidence import group_periods
from .providers import Services
from .research import Investigation, LANGUAGES, new_view, new_work
from .storage import Archive, public_snapshot, recovery, settle


class Previous(BaseModel):
    question: str
    report: str


class Question(BaseModel):
    inputs: str = Field(min_length=1)
    lang: str = "zh-Hans"
    timeline: bool = False
    previous_run_id: str | None = None
    previous_context: Previous | None = None
    resume_run_id: str | None = None
    recovery_action: Literal["continue", "write_report"] | None = None


class Conversation(BaseModel):
    id: str | None = None
    title: str
    data: dict
    skip_timestamp_update: bool = False


def application(database, frontend=None, service_factory=Services):
    active = {}
    current = {}

    @asynccontextmanager
    async def lifespan(app):
        app.state.archive = Archive(database)
        app.state.archive.interrupt()
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=15)) as client:
            app.state.client = client
            try:
                yield
            finally:
                jobs = [task for task in active.values()]
                for job in jobs:
                    job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)
                app.state.archive.db.close()

    app = FastAPI(title="Trace Atlas", lifespan=lifespan)

    def require_run(identifier):
        run = current.get(identifier) or app.state.archive.get(identifier)
        if run is None:
            raise HTTPException(404, "Unknown investigation")
        return run

    @app.post("/solve")
    async def solve(body: Question):
        if not body.inputs.strip() or bool(body.resume_run_id) != bool(body.recovery_action):
            raise HTTPException(422, "Invalid question or recovery request")
        if body.resume_run_id:
            run = require_run(body.resume_run_id)
            if run["id"] in active or run["status"] == "completed":
                raise HTTPException(409, "This investigation is active or already complete")
            if body.recovery_action == "write_report" and not run["work"]["sources"]:
                raise HTTPException(409, "No saved evidence")
            run["attempt"] += 1
            # Interrupted searches may be reattempted; successful evidence remains reusable.
            run["work"]["queries"] = [q for q in run["work"]["queries"] if any(
                t["status"] == "completed" and (run["view"]["task_activity"].get(t["id"], {}).get("query") == q or t.get("query_key") == q)
                for t in run["view"]["tasks"].values())]
            run["work"]["stagnation"] = 0
        else:
            request = body.model_dump(include={"inputs", "lang", "timeline"})
            if body.previous_run_id:
                previous = require_run(body.previous_run_id)
                request["previous"] = {"question": previous["request"]["inputs"], "report": previous["view"]["article"]}
            elif body.previous_context:
                request["previous"] = body.previous_context.model_dump()
            run = {"id": uuid4().hex, "attempt": 1, "request": request, "work": new_work(), "view": new_view()}
        run.update(status="running", error=None)
        current[run["id"]] = run
        app.state.archive.save(run)
        identifier = run["id"]
        stream = asyncio.Queue()
        connected = True

        async def emit(kind, data, task=None):
            if kind not in {"research.progress", "planner.observation"}:
                app.state.archive.save(run)
            if connected:
                stream.put_nowait(json.dumps({"schema_version": 2, "type": kind, "task_id": task, "data": data}, ensure_ascii=False))

        async def execute():
            try:
                await emit("research.run", recovery(run))
                await emit("research.started", {"question": run["request"]["inputs"]})
                if body.resume_run_id:
                    await emit("research.restored", run["view"])
                services = service_factory(app.state.client)
                runner = Investigation(run["request"], run["view"], run["work"], services, services, emit)
                await runner.run(body.recovery_action == "write_report")
                run["status"] = "completed"
                await emit("usage.updated", {"usage": run["view"]["usage"]})
                await emit("research.recovery", recovery(run))
                await emit("research.completed", run["view"])
            except (Exception, asyncio.CancelledError) as exc:
                cancelled = isinstance(exc, asyncio.CancelledError)
                run.update(status="cancelled" if cancelled else "failed", error=None if cancelled else str(exc))
                settle(run)
                await emit("research.recovery", recovery(run))
                await emit("research.cancelled" if cancelled else "research.failed", {"error": run["error"], "tasks": run["view"]["tasks"]})
            finally:
                app.state.archive.save(run)
                active.pop(identifier, None)
                current.pop(identifier, None)
                stream.put_nowait(None)

        active[identifier] = asyncio.create_task(execute())

        async def events():
            nonlocal connected
            try:
                while True:
                    try:
                        value = await asyncio.wait_for(stream.get(), 10)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if value is None:
                        break
                    yield "event: response\ndata: " + value + "\n\n"
            finally:
                connected = False

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"X-Research-Run-ID": identifier, "X-Research-Attempt": str(run["attempt"]),
                                          "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/research/{identifier}")
    async def inspect(identifier: str):
        return public_snapshot(require_run(identifier))

    @app.post("/api/research/{identifier}/stop")
    async def stop(identifier: str):
        run = require_run(identifier)
        if identifier in active:
            run["status"] = "stopping"
            app.state.archive.save(run)
            active[identifier].cancel()
        return recovery(run)

    @app.post("/api/research/{identifier}/timeline-view")
    async def regroup(identifier: str, refresh: bool = False):
        run = require_run(identifier)
        view = run["view"]
        if run["status"] in {"running", "stopping"}:
            raise HTTPException(409, "Investigation is still in progress")
        if refresh or not view.get("timeline_view"):
            result = await group_periods(service_factory(app.state.client), view.get("timeline", []),
                                         LANGUAGES.get(run["request"]["lang"].lower(), "English"), run["request"]["inputs"])
            latest = require_run(identifier)
            if latest["attempt"] != run["attempt"]:
                raise HTTPException(409, "Investigation resumed")
            latest["view"].update(timeline_view=result, timeline_error=False)
            app.state.archive.save(latest)
            return result
        return view["timeline_view"]

    @app.get("/api/conversations")
    async def list_chats(include_empty: bool = True):
        return app.state.archive.chats(include_empty)

    @app.post("/api/conversations", status_code=201)
    async def create(body: Conversation):
        identifier = body.id or uuid4().hex
        try:
            app.state.archive.store_chat(identifier, body.title, body.data, create=True)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Conversation already exists")
        return app.state.archive.chat(identifier)

    @app.get("/api/conversations/{identifier}")
    async def read(identifier: str):
        chat = app.state.archive.chat(identifier)
        if not chat:
            raise HTTPException(404, "Unknown conversation")
        return chat

    @app.put("/api/conversations/{identifier}")
    async def update(identifier: str, body: Conversation):
        if not app.state.archive.store_chat(identifier, body.title, body.data, create=False, keep_time=body.skip_timestamp_update):
            raise HTTPException(404, "Unknown conversation")
        return {"id": identifier}

    @app.delete("/api/conversations/{identifier}")
    async def delete(identifier: str):
        if not app.state.archive.db.execute("DELETE FROM conversations WHERE id=?", (identifier,)).rowcount:
            raise HTTPException(404, "Unknown conversation")
        return {"id": identifier}

    if frontend:
        app.mount("/", StaticFiles(directory=frontend, html=True), name="workbench")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8812)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--database", default=".atlas/workbench.sqlite")
    parser.add_argument("--frontend-dist", type=Path, help="Read-only build of an existing Trace Atlas workbench")
    parser.add_argument("--import-history", type=Path, help="Import saved rounds from a checkpointed v2 workbench database")
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("Environment file does not exist")
        load_dotenv(args.env_file)
    if args.frontend_dist and not (args.frontend_dist / "index.html").is_file():
        parser.error("Frontend build is missing index.html")
    if args.import_history:
        if args.import_history.resolve()==Path(args.database).resolve() or not args.import_history.is_file():
            parser.error("Supply a separate existing history snapshot")
        archive=Archive(args.database)
        try:
            imported,skipped=archive.import_history(args.import_history)
            print(f"Imported {imported} saved conversations. Skipped {skipped} older-format records.", flush=True)
        finally:
            archive.db.close()
    uvicorn.run(application(args.database, args.frontend_dist), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
