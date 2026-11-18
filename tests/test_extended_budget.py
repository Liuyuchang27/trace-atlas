"""High total allowance still enforces admission, cancellation and progress bounds."""

import asyncio
from pathlib import Path

import pytest
from test_adaptive import fixture
from test_adaptive_http import http_scenario

from news_agent.budget import Budget, BudgetExceeded
from news_agent.model_routing import PROFILE, output_limit
from news_agent.resource_control import check_progress, execution_limits
from news_agent.schemas import BudgetConfig
from news_agent.storage import RunStore, read_json

ROOT = Path(__file__).resolve().parents[1]


def config():
    return BudgetConfig.model_validate(read_json(ROOT / "configs/diagnostic_500k.json"))


def test_original_80k_profile_is_unchanged_and_extended_graph_limit_fits():
    assert BudgetConfig().model_dump() == read_json(ROOT / "configs/base.json")
    assert config().max_tokens == 500000
    assert execution_limits(BudgetConfig())["recursion_limit"] == 80
    assert execution_limits(config())["recursion_limit"] > 2 * config().max_model_calls


def test_large_budget_admission_includes_pending_usage_and_retains_writing_reserve(tmp_path):
    b = Budget(config(), RunStore(tmp_path))
    b.state["tokens"] = 37034
    b.refresh_phase(28115)  # actual v17 early-stop estimate
    assert b.state["phase"] == "research"
    b.state["tokens"] = 479000
    b.before_model("writing", 1000)
    assert b.state["phase"] == "writing"
    assert b.used_tokens() == 479000 + 1000 + config().max_output_tokens
    recovered = Budget(config(), b.store)
    assert recovered.used_tokens() == 479000 + 1000 + config().max_output_tokens
    with pytest.raises(BudgetExceeded, match="token_limit"):
        recovered.before_model("too-large", 20000)
    assert recovered.state["model_calls"] == 1


def test_larger_model_allowance_still_stops_at_64_requests(tmp_path):
    b = Budget(config(), RunStore(tmp_path))
    for i in range(64):
        b.before_model(str(i), 1)
        b.after_model(str(i), 10)
    with pytest.raises(BudgetExceeded, match="model_call_limit"):
        b.before_model("extra", 1)
    assert b.state["tokens"] == 640


def test_no_progress_warning_survives_restart_without_forced_cutoff(tmp_path):
    _, b, _, a = fixture(tmp_path)
    b.config = config()
    check_progress(a, b)
    for _ in range(3):
        check_progress(a, b)
    assert b.state["phase"] == "research"
    b = Budget(config(), b.store)
    for _ in range(3):
        check_progress(a, b)
    assert b.state["phase"] == "research"
    assert b.state["stop_reason"] is None
    assert "extended_no_progress_warning" in (b.store.root / "trajectory.jsonl").read_text()
    assert b.state["model_calls"] == 0


def test_new_evidence_resets_only_stagnation_not_cumulative_budget(tmp_path):
    _, b, _, a = fixture(tmp_path)
    b.config = config()
    check_progress(a, b)
    for _ in range(5):
        check_progress(a, b)
    b.state["delivered_passages"] = ["fixture-new-original"]
    b.state["tokens"] = 12345
    check_progress(a, b)
    assert b.state["extended_progress"]["stalled"] == 0
    assert b.state["tokens"] == 12345


def test_actual_sdk_high_budget_uses_staged_models_and_stops_without_spending_allowance(
    tmp_path, monkeypatch
):
    result, requests, web = asyncio.run(
        http_scenario(
            tmp_path,
            monkeypatch,
            model_profile=PROFILE,
            budget_override=config(),
        )
    )
    assert result["model_finalized"]
    assert result["budget"]["model_calls"] == len(requests) == 11
    assert result["budget"]["tokens"] == 3850
    assert len(web) == 4
    effective = read_json(tmp_path / "effective_config.json")
    assert effective["budget"] == config().model_dump()
    assert effective["execution_limits"] == execution_limits(config())
    assert requests[0]["max_tokens"] == 10000
    assert requests[-1]["max_tokens"] == 16000
    assert {r["max_tokens"] for r in requests} == {6000, 10000, 12000, 16000}
    assert output_limit("report_review", config()) == 16000
