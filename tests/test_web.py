import asyncio
import json
import shutil
import threading
import time
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import news_agent.harness as harness
from news_agent.budget import Budget, mock_usage
from news_agent.mock_model import FixtureChatModel
from news_agent.progress import document_locator, plan_tool
from news_agent.schemas import BudgetConfig, DocumentVersion
from news_agent.storage import EvidenceStore, RunStore, read_json
from news_agent.web import RunManager, RunRequest, make_server
from news_agent.web_view import build_snapshot, report_html, safe_url


def test_public_plan_is_atomic_evidence_bound_and_acyclic(tmp_path):
    store = RunStore(tmp_path)
    evidence = EvidenceStore(store)
    budget = Budget(BudgetConfig(), store)
    doc = DocumentVersion(version_id="one", title="One", url="https://example.com", content="Text")
    passage = evidence.add_document(doc)[0]
    tool = plan_tool(evidence, budget)

    def call(steps):
        return json.loads(asyncio.run(tool.ainvoke({"steps": steps})))

    step = {"step_id": "initial", "title": "Inspect evidence", "status": "in_progress"}
    assert "saved_steps" in call([step])
    before = (tmp_path / "public_plan.json").read_bytes()
    bad = {"step_id": "gap", "title": "Find a dated record", "passage_ids": [passage.passage_id]}
    assert "error" in call([bad])  # saved document alone is not a delivered passage
    assert (tmp_path / "public_plan.json").read_bytes() == before
    budget.state["delivered_passages"] = [passage.passage_id]
    assert "saved_steps" in call([{**bad, "parent_id": "initial", "kind": "gap"}])
    before = (tmp_path / "public_plan.json").read_bytes()
    assert "error" in call([{**step, "parent_id": "gap"}])
    assert (tmp_path / "public_plan.json").read_bytes() == before
    assert "error" in call([{**step, "parent_id": "missing"}])


class PublicPlanFixture(FixtureChatModel):
    initial_published: bool = False
    gap_published: bool = False
    followup_sent: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        names = {t["function"]["name"] for t in kwargs.get("tools", [])}
        action = None
        if "publish_plan" in names:
            if not self.initial_published:
                self.initial_published = True
                action = (
                    "publish_plan",
                    {
                        "steps": [
                            {
                                "step_id": "status",
                                "title": "Find current transaction status",
                                "status": "in_progress",
                            }
                        ]
                    },
                )
            elif not self.gap_published:
                reads = [
                    json.loads(m.content)
                    for m in messages
                    if isinstance(m, ToolMessage) and '"passages"' in m.content
                ]
                if not reads:
                    for message in messages:
                        if message.type == "human" and message.text.startswith("Here is a summary"):
                            packet = json.loads(message.text.split("\n\n", 1)[1])
                            for doc in packet["documents"]:
                                reads.append(
                                    {
                                        "passages": [
                                            {"passage_id": pid, "text": text}
                                            for pid, text in doc["excerpts"].items()
                                        ]
                                    }
                                )
                if reads and reads[0]["passages"]:
                    self.gap_published = True
                    action = (
                        "publish_plan",
                        {
                            "steps": [
                                {
                                    "step_id": "closing-gap",
                                    "parent_id": "status",
                                    "kind": "gap",
                                    "title": "Completion evidence missing",
                                    "status": "blocked",
                                    "reason": (
                                        "The passage announces a plan, not observed completion."
                                    ),
                                    "passage_ids": [reads[0]["passages"][0]["passage_id"]],
                                },
                                {
                                    "step_id": "follow-up",
                                    "parent_id": "closing-gap",
                                    "kind": "followup",
                                    "title": "Seek a completion record",
                                    "status": "in_progress",
                                },
                            ]
                        },
                    )
            elif not self.followup_sent and "search" in names:
                self.followup_sent = True
                action = (
                    "search",
                    {
                        "query": "Aurora Birch closing completion",
                        "purpose": "Find evidence missing from the announcement.",
                        "intent": "gap",
                        "step_id": "follow-up",
                    },
                )
        if action:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": action[0],
                        "args": action[1],
                        "id": (
                            f"public-{self.initial_published}"
                            f"-{self.gap_published}-{self.followup_sent}"
                        ),
                    }
                ],
            )
            message.usage_metadata = mock_usage(messages, str(message.tool_calls))
            return ChatResult(generations=[ChatGeneration(message=message)])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_real_sdk_plan_gap_followup_projection_and_safe_report(
    tmp_path, task, corpus_path, monkeypatch
):
    monkeypatch.setattr(harness, "FixtureChatModel", PublicPlanFixture)
    result = asyncio.run(
        harness.run_research(
            tmp_path, task, corpus_path, BudgetConfig(), mock=True, visual_progress=True
        )
    )
    assert result["model_finalized"], result
    snapshot = build_snapshot(tmp_path)
    plans = {n["id"]: n for n in snapshot["nodes"] if n["provenance"] == "model_reported"}
    assert {"plan:status", "plan:closing-gap", "plan:follow-up"} <= plans.keys()
    followup = next(
        n
        for n in snapshot["nodes"]
        if n.get("tool") == "search"
        and n["details"]["arguments"]["query"] == "Aurora Birch closing completion"
    )
    assert {
        "from": "plan:follow-up",
        "to": followup["id"],
        "kind": "context",
        "sequence": followup["sequence"],
    } in snapshot["edges"]
    assert plans["plan:closing-gap"]["updates"][0]["passage_ids"]
    gap = next(n for n in snapshot["nodes"] if n["id"].startswith("intent:") and n["kind"] == "gap")
    assert any(
        e["from"] == gap["id"] and e["to"] == followup["id"] and e["kind"] == "explicit"
        for e in snapshot["edges"]
    )
    assert all(row["date"] != "2026-06-12" for row in snapshot["timeline"])
    assert snapshot["timeline"] and all(
        t["citations"] or "无法确定" in t["text"] for t in snapshot["timeline"]
    )
    snapshot["question"] = '<script>alert("xss")</script>'
    snapshot["report"] += '\n<img src="x" onerror="alert(1)">'
    exported = report_html(snapshot)
    assert "<script>" not in exported and '<img src="x"' not in exported
    assert "&lt;script&gt;" in exported and 'href="#p-' in exported
    assert "合成资料" in exported and "带证据的事件时间线" in exported
    assert safe_url("javascript:alert(1)") is None
    assert safe_url("https://user:secret@example.com") is None


def test_legacy_trace_never_invents_a_plan_or_a_gap(tmp_path):
    store = RunStore(tmp_path)
    store.save("task.json", {"question": "Question", "mode": "retrospective"})
    store.event(
        "tool_dispatch", action_id="one", name="search", arguments={"query": "official record"}
    )
    store.event("tool_result", action_id="one", name="search")
    snapshot = build_snapshot(tmp_path)
    assert len(snapshot["nodes"]) == 2
    assert not any(n["provenance"] == "model_reported" for n in snapshot["nodes"])
    assert snapshot["timeline"] == []  # timestamps in journal are not event dates


def test_saved_text_locator_exposes_exact_body_without_network(tmp_path):
    evidence = EvidenceStore(RunStore(tmp_path))
    budget = Budget(BudgetConfig(), evidence.store)
    text = "Navigation and cookie controls. " * 700 + "\n\nThe transaction completed on 2023-10-13."
    doc = DocumentVersion(
        version_id="selected", title="Record", url="https://example.com", content=text
    )
    evidence.add_document(doc)
    locator = document_locator(evidence, budget)
    result = json.loads(
        asyncio.run(locator.ainvoke({"version_id": "selected", "query": "transaction completed"}))
    )
    assert result["passages"][0]["text"] == "The transaction completed on 2023-10-13."
    passage = result["passages"][0]
    assert doc.content[passage["start"] : passage["end"]] == passage["text"]
    assert passage["passage_id"] in budget.state["delivered_passages"]
    assert budget.state.get("retrieval_api_calls", 0) == 0
    assert evidence.documents["selected"].content == text
    assert "error" in json.loads(
        asyncio.run(locator.ainvoke({"version_id": "unselected", "query": "text"}))
    )
    budget.finish_research("test-stop")
    assert "research_finished" in asyncio.run(
        locator.ainvoke({"version_id": "selected", "query": "completed"})
    )


def test_cancelled_visual_run_saves_partial_report_and_unknown_reservation(
    tmp_path, task, corpus_path, monkeypatch
):
    class SlowModel(FixtureChatModel):
        async def _agenerate(self, *args, **kwargs):
            await asyncio.sleep(10)
            return super()._generate(*args, **kwargs)

    monkeypatch.setattr(harness, "FixtureChatModel", SlowModel)

    async def scenario():
        job = asyncio.create_task(
            harness.run_research(
                tmp_path, task, corpus_path, BudgetConfig(), mock=True, visual_progress=True
            )
        )
        for _ in range(100):
            if (tmp_path / "budget.json").exists() and read_json(tmp_path / "budget.json")[
                "pending"
            ]:
                break
            await asyncio.sleep(0.01)
        job.cancel()
        return await job

    result = asyncio.run(scenario())
    assert result["status"] == "partial" and result["stop_reason"] == "cancelled_by_user"
    assert not result["budget"]["pending"] and result["budget"]["unknown_usage_calls"] == 1
    assert Path(result["report"]).exists()


@pytest.fixture
def local_server(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/fixtures"
    shutil.copytree(source, tmp_path / "data/fixtures")
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.manager.close()
    server.shutdown()
    server.server_close()
    for process in getattr(server.manager, "processes", {}).values():
        process.wait(timeout=10)


def test_web_api_origin_boundary_and_real_offline_worker(local_server):
    server, url = local_server
    with httpx.Client(base_url=url, trust_env=False) as client:
        health = client.get("/api/health").json()
        token = health["csrf_token"]
        assert client.get("/").status_code == 200
        assert client.get("/.env").status_code == 404
        assert client.get("/api/runs/not-a-path").status_code == 404
        assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 403
        assert client.post("/api/runs", json={"mode": "fixture"}).status_code == 403
        headers = {"X-Research-Token": token, "Origin": url}
        assert (
            client.post(
                "/api/runs",
                json={"mode": "fixture"},
                headers={**headers, "Origin": "https://attacker.example"},
            ).status_code
            == 403
        )
        start = client.post("/api/runs", json={"mode": "fixture"}, headers=headers)
        assert start.status_code == 202, start.text
        key = start.json()["id"]
        assert (
            client.post("/api/runs", json={"mode": "fixture"}, headers=headers).status_code == 409
        )
        snapshot = {}
        for _ in range(150):
            snapshot = client.get("/api/runs/" + key).json()
            if not server.manager.process_active(key):
                break
            time.sleep(0.05)
        snapshot = client.get("/api/runs/" + key).json()
        assert snapshot["status"] in {"complete", "partial"}, snapshot
        assert snapshot["mock"] is True and snapshot["sources"]
        assert snapshot["budget"].get("retrieval_api_calls", 0) == 0
        assert "csrf_token" not in snapshot
        exported = client.get("/api/runs/" + key + "/report.html")
        assert exported.status_code == 200 and 'href="#p-' in exported.text
        assert "attachment" in exported.headers["Content-Disposition"]
        assert (
            client.get("/api/runs/" + key + "/trace.json").json()["run_id"]
            == start.json()["run_id"]
        )


def test_run_validation_does_not_start_paid_request(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    manager = RunManager(tmp_path)
    with pytest.raises(ValueError, match="请输入"):
        manager.start(RunRequest(question=""))
    with pytest.raises(ValueError, match=".env"):
        manager.start(RunRequest(question="test"))
    assert not manager.jobs and not (tmp_path / "runs").exists()
