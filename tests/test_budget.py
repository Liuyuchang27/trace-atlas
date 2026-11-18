import time

import pytest
from langchain_core.messages import HumanMessage

from news_agent.budget import Budget, BudgetCallback, BudgetExceeded
from news_agent.schemas import BudgetConfig
from news_agent.storage import RunStore


def test_unknown_model_consumption_is_reserved_across_restart(tmp_path):
    config = BudgetConfig()
    original = Budget(config, RunStore(tmp_path))
    original.before_model("crashed-request", 1234)
    restarted = Budget(config, RunStore(tmp_path))
    assert restarted.used_tokens() == 1234 + config.max_output_tokens
    assert restarted.state["model_calls"] == 1
    assert restarted.state["deadline_at"] == original.state["deadline_at"]
    restarted.after_model("crashed-request", None, error=True)
    assert restarted.state["unknown_usage_calls"] == 1
    assert restarted.used_tokens() == original.used_tokens()


def test_writing_reserve_is_not_extra_budget(tmp_path):
    config = BudgetConfig(max_model_calls=3, reserve_model_calls=1)
    budget = Budget(config, RunStore(tmp_path))
    for i in range(3):
        budget.before_model(str(i), 1)
        budget.after_model(str(i), 10)
    assert budget.state["phase"] == "writing"
    with pytest.raises(BudgetExceeded, match="model_call_limit"):
        budget.before_model("fourth", 1)
    assert budget.state["model_calls"] == 3


def test_deadline_does_not_reset_on_recovery(tmp_path):
    budget = Budget(BudgetConfig(), RunStore(tmp_path))
    budget.state["deadline_at"] = time.time() - 1
    budget.save()
    recovered = Budget(BudgetConfig(), RunStore(tmp_path))
    with pytest.raises(BudgetExceeded, match="deadline_reached"):
        recovered.before_model("late", 1)


def test_only_actual_request_tools_are_reserved_with_conservative_fallback(tmp_path):
    budget = Budget(BudgetConfig(), RunStore(tmp_path))
    budget.state["tokens"] = 70000
    callback = BudgetCallback(budget, 40000)
    messages = [[HumanMessage("A short evidence assessment.")]]
    callback.on_chat_model_start({}, messages, run_id="empty-tools", invocation_params={})
    assert budget.state["model_calls"] == 1
    assert budget.state["pending"]["empty-tools"] < 10000
    budget.after_model("empty-tools", 100)
    with pytest.raises(BudgetExceeded, match="token_limit"):
        callback.on_chat_model_start(
            {},
            messages,
            run_id="large-tools",
            invocation_params={"tools": [{"schema": "x" * 15000}]},
        )
    with pytest.raises(BudgetExceeded, match="token_limit"):
        callback.on_chat_model_start({}, messages, run_id="unknown-request-shape")
    assert budget.state["model_calls"] == 1
