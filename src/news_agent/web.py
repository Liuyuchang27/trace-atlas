"""Loopback-only test application for the existing research worker."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import Field, ValidationError

from .schemas import BudgetConfig, StrictModel
from .storage import write_json
from .web_view import build_snapshot, maybe_json, report_html, run_key


class RunRequest(StrictModel):
    question: str = Field(default="", max_length=3000)
    condition: str = "baseline"
    mode: str = "live"
    workflow: str = "adaptive-event-map"
    model_profile: str = "auto"
    parent_run_id: str | None = None


class RunManager:
    def __init__(self, root, budget=None):
        self.root = root.resolve()
        self.lock = threading.Lock()
        self.jobs = {}
        self.live_budget = budget or BudgetConfig()

    def catalog(self):
        paths = []
        for base in [self.root / "runs/web", self.root / "runs/live-web"]:
            if base.exists():
                paths.extend(
                    p.parent
                    for p in base.rglob("effective_config.json")
                    if "code_snapshot" not in p.parts
                )
        paths.extend(self.jobs.values())
        demo = self.root / "examples/h0_demo"
        if demo.exists():
            paths.append(demo)
        return {run_key(self.root, p): p for p in set(paths)}

    def process_active(self, key):
        entry = getattr(self, "processes", {}).get(key)
        return entry is not None and entry.poll() is None

    def list_runs(self):
        rows = []
        for key, path in self.catalog().items():
            task = maybe_json(path / "task.json")
            request = maybe_json(path / "web_request.json")
            outcome = maybe_json(path / "run.json")
            config = maybe_json(path / "effective_config.json")
            active = self.process_active(key)
            rows.append(
                {
                    "id": key,
                    "run_id": path.name,
                    "question": task.get("question", request.get("question", "离线集成样例")).split(
                        "\n\nRun observation context:"
                    )[0],
                    "condition": config.get("condition", request.get("condition", "baseline")),
                    "workflow": config.get("workflow", request.get("workflow", "legacy")),
                    "mock": config.get("mock", request.get("mode") == "fixture"),
                    "status": "running" if active else outcome.get("status", "interrupted"),
                    "read_only": key not in self.jobs,
                    "started_at": outcome.get("budget", {}).get("deadline_at", 0)
                    - config.get("budget", {}).get("deadline_seconds", 600)
                    if outcome.get("budget", {}).get("deadline_at")
                    else (path / "effective_config.json").stat().st_mtime
                    if (path / "effective_config.json").exists()
                    else path.stat().st_mtime,
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                row["status"] == "running",
                row["started_at"],
            ),
            reverse=True,
        )[:50]

    def snapshot(self, key):
        path = self.catalog().get(key)
        if path is None:
            raise ValueError("运行不存在")
        status = maybe_json(path / "worker_status.json").get("status")
        data = build_snapshot(path, active=self.process_active(key), worker_status=status)
        data["id"] = key
        data["can_cancel"] = key in self.jobs and self.process_active(key)
        return data

    def start(self, request):
        if request.model_profile not in {"auto", "single", "staged-deepseek-v1"}:
            raise ValueError("未知模型配置")
        if request.workflow not in {"adaptive-event-map", "legacy"}:
            raise ValueError("未知调查流程")
        if request.workflow == "adaptive-event-map" and request.condition != "baseline":
            raise ValueError("事件调查使用通用流程；T/S仅用于旧实验模式")
        if request.mode not in {"live", "fixture"}:
            raise ValueError("请选择联网调查或离线合成验收")
        if request.condition not in {"baseline", "s", "t", "ts"}:
            raise ValueError("未知实验条件")
        if request.mode == "live":
            if not request.question.strip():
                raise ValueError("请输入研究问题")
            if not all(
                os.environ.get(k)
                for k in ["TAVILY_API_KEY", "DEEPSEEK_API_KEY", "NEWS_AGENT_MODEL"]
            ):
                raise ValueError("请在本地.env填写DeepSeek模型与密钥、Tavily密钥后重启服务")
            if request.condition in {"t", "ts"} and not any(
                s in request.question.lower() for s in ["收购", "acquisit", "acquir", "merger"]
            ):
                raise ValueError("T/TS目前仅支持收购事件；本问题请选择baseline或S")
        if request.parent_run_id and request.parent_run_id not in self.catalog():
            raise ValueError("找不到所基于的运行")
        with self.lock:
            if any(self.process_active(k) for k in self.jobs):
                raise RuntimeError("已有任务运行中，请等待或停止后再发起，避免重复付费")
            rid = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
            path = self.root / "runs/web" / rid
            path.mkdir(parents=True)
            body = request.model_dump()
            if request.parent_run_id:
                parent = self.snapshot(request.parent_run_id)
                body["question"] = (
                    "上一请求："
                    + parent["question"][:1500]
                    + "\n用户后续请求或条件变更："
                    + request.question
                )
            write_json(path / "web_request.json", body)
            budget = self.live_budget if request.mode == "live" else BudgetConfig()
            write_json(path / "requested_budget.json", budget.model_dump(mode="json"))
            key = run_key(self.root, path)
            env = os.environ.copy()
            hosts = env.get("NO_PROXY", "").split(",")
            env["NO_PROXY"] = ",".join(
                dict.fromkeys(hosts + ["api.deepseek.com", "api.tavily.com"])
            )
            with (path / "worker.log").open("w") as output:
                process = subprocess.Popen(
                    [sys.executable, "-m", "news_agent.web_worker", str(self.root), str(path)],
                    cwd=self.root,
                    env=env,
                    stdout=output,
                    stderr=output,
                )
            self.jobs[key] = path
            if not hasattr(self, "processes"):
                self.processes = {}
            self.processes[key] = process
            return {"id": key, "run_id": rid}

    def cancel(self, key):
        with self.lock:
            if not self.process_active(key):
                raise ValueError("此任务未在当前服务中运行")
            self.processes[key].terminate()
            return {"status": "cancellation_requested"}

    def close(self):
        for key in self.jobs:
            if self.process_active(key):
                self.processes[key].terminate()


def make_server(root, port=8768, *, budget=None):
    manager = RunManager(Path(root), budget)
    token = secrets.token_urlsafe(32)
    assets = Path(__file__).with_name("web_assets")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # do not log questions or URL parameters to console

        def respond(
            self, payload, status=200, content_type="application/json; charset=utf-8", download=None
        ):
            data = (
                payload.encode()
                if isinstance(payload, str)
                else json.dumps(payload, ensure_ascii=False).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="' + download + '"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def valid_host(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def do_GET(self):
            if not self.valid_host():
                return self.respond({"error": "invalid host"}, 403)
            path = urlsplit(self.path).path
            if path in {"/", "/app.js", "/style.css", "/icons.js"}:
                name = {
                    "/": "index.html",
                    "/app.js": "app.js",
                    "/style.css": "style.css",
                    "/icons.js": "icons.js",
                }[path]
                mime = {
                    "index.html": "text/html",
                    "app.js": "text/javascript",
                    "icons.js": "text/javascript",
                    "style.css": "text/css",
                }[name]
                return self.respond(
                    (assets / name).read_text(), content_type=mime + "; charset=utf-8"
                )
            if path == "/api/health":
                return self.respond(
                    {
                        "ready": all(
                            os.environ.get(k)
                            for k in ["TAVILY_API_KEY", "DEEPSEEK_API_KEY", "NEWS_AGENT_MODEL"]
                        ),
                        "model": os.environ.get("NEWS_AGENT_MODEL", "未配置"),
                        "csrf_token": token,
                        "local_only": True,
                        "live_budget": manager.live_budget.model_dump(mode="json"),
                        "busy": any(manager.process_active(k) for k in manager.jobs),
                    }
                )
            if path == "/api/runs":
                return self.respond({"runs": manager.list_runs()})
            segments = path.strip("/").split("/")
            if len(segments) in {3, 4} and segments[:2] == ["api", "runs"]:
                try:
                    snapshot = manager.snapshot(segments[2])
                except ValueError as exc:
                    return self.respond({"error": str(exc)}, 404)
                if len(segments) == 3:
                    return self.respond(snapshot)
                export = segments[3]
                if export == "report.html":
                    return self.respond(
                        report_html(snapshot),
                        content_type="text/html; charset=utf-8",
                        download="research-report.html",
                    )
                if export == "trace.json":
                    return self.respond(
                        json.dumps(snapshot, ensure_ascii=False, indent=2),
                        download="research-trace.json",
                    )
                if export == "question-state.json":
                    return self.respond(snapshot.get("question_state", {}))
                if export == "event-graph.json":
                    return self.respond(snapshot.get("event_graph", {}))
            return self.respond({"error": "not found"}, 404)

        def do_POST(self):
            expected = f"http://{self.headers.get('Host')}"
            if not self.valid_host() or self.headers.get("Origin", expected) != expected:
                return self.respond({"error": "same-origin required"}, 403)
            if not hmac.compare_digest(self.headers.get("X-Research-Token", ""), token):
                return self.respond({"error": "request token required"}, 403)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 24000:
                    raise ValueError("请求大小无效")
                payload = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                if path == "/api/runs":
                    return self.respond(manager.start(RunRequest.model_validate(payload)), 202)
                segments = path.strip("/").split("/")
                if (
                    len(segments) == 4
                    and segments[:2] == ["api", "runs"]
                    and segments[3] == "cancel"
                ):
                    return self.respond(manager.cancel(segments[2]))
                return self.respond({"error": "not found"}, 404)
            except RuntimeError as exc:
                return self.respond({"error": str(exc)}, 409)
            except ValidationError:
                return self.respond({"error": "请求格式无效或问题过长"}, 400)
            except ValueError as exc:
                if isinstance(exc, json.JSONDecodeError):
                    return self.respond({"error": "请求必须是有效JSON"}, 400)
                return self.respond({"error": str(exc)}, 400)
            except (TypeError, KeyError):
                return self.respond(
                    {"error": "请求无效：检查问题、条件、当前运行和本地模型配置"}, 400
                )

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.manager = manager
    return server


def serve(root, port=8768, *, budget=None):
    server = make_server(root, port, budget=budget)
    print(f"Agent test app: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.manager.close()
        server.server_close()
