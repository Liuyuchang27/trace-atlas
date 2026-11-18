"""Explicit no-total-budget diagnostics still account every request and uncertainty."""

import asyncio
import json
import math
from pathlib import Path

import pytest
from test_adaptive_http import http_scenario

from news_agent.budget import Budget, BudgetExceeded
from news_agent.model_routing import PROFILE
from news_agent.schemas import BudgetConfig
from news_agent.storage import RunStore, read_json

ROOT = Path(__file__).resolve().parents[1]


def config():
    return BudgetConfig.model_validate(read_json(ROOT / "configs/diagnostic_uncapped.json"))


def test_uncapped_crosses_previous_totals_and_preserves_ledger_on_resume(tmp_path):
    b = Budget(config(), RunStore(tmp_path))
    b.state.update(tokens=500000, model_calls=64, tool_calls=96)
    b.save()
    b.before_model("beyond", 1000000)
    b.before_tool("read_document")
    assert b.state["phase"] == "research"
    assert b.used_tokens() == 1516000
    b = Budget(config(), b.store)
    assert b.state["model_calls"] == 65 and b.state["tool_calls"] == 97
    assert b.timeout() is None and math.isinf(b.remaining_time())
    b.after_model("beyond", 21000)
    assert b.used_tokens() == 521000
    assert b.state["phase"] == "research" and b.state["stop_reason"] is None
    json.dumps(b.state, allow_nan=False)


def test_uncapped_unknown_request_keeps_reservation_without_rebilling(tmp_path):
    b = Budget(config(), RunStore(tmp_path))
    b.before_model("unknown", 100)
    b.settle_pending()
    b.settle_pending()
    assert b.used_tokens() == 16100
    assert b.state["model_calls"] == b.state["unknown_usage_calls"] == 1


def test_uncapped_tokens_does_not_disable_explicit_other_limits(tmp_path):
    c = config().model_copy(update={"max_model_calls": 7})
    b = Budget(c, RunStore(tmp_path))
    b.state["model_calls"] = 7
    with pytest.raises(BudgetExceeded, match="model_call_limit"):
        b.before_model("denied", 10)
    assert b.state["stop_reason"] == "research_budget_exhausted"


def test_uncapped_run_remains_cancellable():
    async def check():
        async def worker():
            async with asyncio.timeout(None):
                await asyncio.Event().wait()

        job = asyncio.create_task(worker())
        await asyncio.sleep(0)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job

    asyncio.run(check())


def test_actual_sdk_uncapped_natural_stop_and_staged_outputs(tmp_path, monkeypatch):
    result, requests, web = asyncio.run(
        http_scenario(tmp_path, monkeypatch, model_profile=PROFILE, budget_override=config())
    )
    assert result["model_finalized"]
    assert result["budget"]["model_calls"] == len(requests) == 11
    assert result["budget"]["tokens"] == 3850 and len(web) == 4
    assert result["token_overshoot"] is None
    assert result["budget"]["deadline_at"] is None
    assert result["stop_reason"] != "research_budget_exhausted"
    assert {r["max_tokens"] for r in requests} == {6000, 10000, 12000, 16000}
    json.dumps(result, allow_nan=False)
