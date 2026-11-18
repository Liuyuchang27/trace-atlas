"""Persistent request accounting, including SDK-internal model calls."""

from __future__ import annotations

import json
import math
import time
from threading import RLock

from langchain_core.callbacks import BaseCallbackHandler

from .schemas import BudgetConfig
from .storage import RunStore, read_json


class BudgetExceeded(RuntimeError):
    pass


def model_error_details(error):
    causes = []
    cause = error.__cause__
    while cause is not None and len(causes) < 6:
        causes.append(type(cause).__name__)
        cause = cause.__cause__
    status = getattr(error, "status_code", None)
    return {
        "error_type": type(error).__name__,
        "cause_types": causes,
        "http_status": status if isinstance(status, int) else None,
    }


def input_upper_estimate(messages, tool_bytes: int = 0) -> int:
    # Conservative byte-based estimate, not a provider tokenizer measurement.
    return sum(len(str(m.content).encode()) + 256 for m in messages) + tool_bytes + 1024


class Budget:
    def __init__(self, config: BudgetConfig, store: RunStore):
        self.config, self.store = config, store
        self._lock = RLock()
        path = store.root / "budget.json"
        self.state = (
            read_json(path)
            if path.exists()
            else {
                "model_calls": 0,
                "tool_calls": 0,
                "tokens": 0,
                "pending": {},
                "deadline_at": time.time() + config.deadline_seconds
                if config.deadline_seconds is not None
                else None,
                "phase": "research",
                "stop_reason": None,
                "search_rounds": 0,
                "unknown_usage_calls": 0,
            }
        )
        self.save()

    def save(self):
        self.store.save("budget.json", self.state)

    def remaining_time(self) -> float:
        deadline = self.state["deadline_at"]
        return max(0.0, deadline - time.time()) if deadline is not None else math.inf

    def timeout(self) -> float | None:
        """None is asyncio's explicit no-total-deadline mode, never a fake large value."""
        return self.remaining_time() if self.state["deadline_at"] is not None else None

    def used_tokens(self) -> int:
        return self.state["tokens"] + sum(self.state["pending"].values())

    def finish_research(self, reason: str):
        with self._lock:
            if self.state["phase"] == "research":
                self.state.update(phase="writing", stop_reason=reason)
                self.save()
                self.store.event("research_stopped", reason=reason)

    def refresh_phase(self, estimate: int = 0):
        c, s = self.config, self.state
        triggers = {
            "model_calls": c.max_model_calls is not None
            and s["model_calls"] >= c.max_model_calls - c.reserve_model_calls,
            "tool_calls": c.max_tool_calls is not None
            and s["tool_calls"] >= c.max_tool_calls - c.reserve_tool_calls,
            "tokens": c.max_tokens is not None
            and self.used_tokens() + estimate + c.max_output_tokens
            > c.max_tokens - c.reserve_tokens,
            "time": self.remaining_time() <= c.reserve_seconds,
        }
        if any(triggers.values()):
            if s["phase"] == "research":
                self.store.event(
                    "research_budget_boundary",
                    used_tokens=self.used_tokens(),
                    next_input_upper_estimate=estimate,
                    next_output_reserve=c.max_output_tokens,
                    research_token_ceiling=c.max_tokens - c.reserve_tokens
                    if c.max_tokens is not None
                    else None,
                    model_calls=s["model_calls"],
                    tool_calls=s["tool_calls"],
                    remaining_seconds=self.timeout(),
                    triggers=triggers,
                )
            self.finish_research("research_budget_exhausted")

    def reopen_temporal_research(self):
        """Re-enter evidence work after natural closure, without resetting any resource ledger."""
        with self._lock:
            if self.state.get("pending") or self.state.get("retrieval_pending"):
                return False
            if self.state["phase"] != "research" and self.state.get("stop_reason") not in {
                "adaptive_gaps_resolved_or_stalled",
                "temporal_reaudit_requested",
            }:
                return False
            self.state.update(phase="research", stop_reason=None)
            self.refresh_phase()
            self.save()
            if self.state["phase"] == "research":
                self.store.event(
                    "research_reopened", reason="temporal_audit_followups", usage_reset=False
                )
                return True
            return False

    def before_model(self, action_id: str, estimate: int):
        with self._lock:
            self.refresh_phase(estimate)
            if self.remaining_time() <= 0:
                raise BudgetExceeded("deadline_reached")
            if (
                self.config.max_model_calls is not None
                and self.state["model_calls"] >= self.config.max_model_calls
            ):
                raise BudgetExceeded("model_call_limit")
            reservation = estimate + self.config.max_output_tokens
            if (
                self.config.max_tokens is not None
                and self.used_tokens() + reservation > self.config.max_tokens
            ):
                raise BudgetExceeded("token_limit")
            self.state["model_calls"] += 1
            if self.state["phase"] == "writing":
                self.state["writing_model_calls"] = self.state.get("writing_model_calls", 0) + 1
            self.state["pending"][action_id] = reservation
            self.save()
            self.store.event(
                "model_dispatch",
                action_id=action_id,
                reserved_tokens=reservation,
                phase=self.state["phase"],
            )

    def after_model(self, action_id: str, tokens: int | None, error: bool = False):
        with self._lock:
            reservation = self.state["pending"].pop(action_id, None)
            if reservation is None:
                return
            accounted = reservation if tokens is None else tokens
            self.state["tokens"] += accounted
            if tokens is None:
                self.state["unknown_usage_calls"] += 1
            self.save()
            self.store.event(
                "model_result",
                action_id=action_id,
                tokens=tokens,
                accounted_tokens=accounted,
                usage_known=tokens is not None,
                error=error,
            )

    def before_tool(self, name: str):
        with self._lock:
            self.refresh_phase()
            if self.remaining_time() <= 0:
                raise BudgetExceeded("deadline_reached")
            if self.state["phase"] != "research":
                raise BudgetExceeded("research_finished")
            if (
                self.config.max_tool_calls is not None
                and self.state["tool_calls"] >= self.config.max_tool_calls
            ):
                raise BudgetExceeded("tool_call_limit")
            round_limit = self.state.get("search_round_limit", 6)
            if (
                name == "search"
                and round_limit is not None
                and self.state["search_rounds"] >= round_limit
            ):
                self.finish_research("round_limit")
                raise BudgetExceeded("round_limit")
            self.state["tool_calls"] += 1
            if name == "search":
                self.state["search_rounds"] += 1
            self.save()

    def settle_pending(self):
        """Cancellation/crash may skip on_llm_error; retain the entire reservation."""
        for action_id in list(self.state["pending"]):
            self.after_model(action_id, None, error=True)


class BudgetCallback(BaseCallbackHandler):
    """Callbacks propagate to SDK summarization, unlike only wrapping the outer agent."""

    raise_error = True
    run_inline = True

    def __init__(self, budget: Budget, tool_bytes: int):
        self.budget, self.tool_bytes = budget, tool_bytes

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        params = kwargs.get("invocation_params")
        # LangChain exposes the actual bound request. Auxiliary/final requests
        # must not reserve schemas for tools that are absent from that request.
        tool_bytes = self.tool_bytes
        if isinstance(params, dict):
            tools = params.get("tools") or []
            if isinstance(tools, list) and all(isinstance(t, dict) for t in tools):
                tool_bytes = len(json.dumps(tools).encode()) + 2048 if tools else 0
        estimate = sum(input_upper_estimate(batch, tool_bytes) for batch in messages)
        self.budget.store.event("request_shape", tool_schema_bytes=tool_bytes)
        self.budget.before_model(str(run_id), estimate)
        if isinstance(params, dict):
            self.budget.store.event(
                "provider_request_config",
                action_id=str(run_id),
                model=params.get("model", params.get("model_name")),
                thinking=(params.get("extra_body") or {}).get("thinking"),
                reasoning_effort=params.get("reasoning_effort"),
            )
        packet = self.budget.state.get("active_context_packet")
        if packet and any(
            packet["packet_id"] in str(m.content) for batch in messages for m in batch
        ):
            self.budget.store.event("model_context_used", action_id=str(run_id), **packet)

    def on_llm_end(self, response, *, run_id, **kwargs):
        usages = [
            getattr(g.message, "usage_metadata", None)
            for batch in response.generations
            for g in batch
            if hasattr(g, "message")
        ]
        tokens = (
            sum(u["total_tokens"] for u in usages)
            if usages and all(u is not None for u in usages)
            else None
        )
        self.budget.after_model(str(run_id), tokens)
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                if message is not None:
                    metadata = message.response_metadata
                    self.budget.store.event(
                        "provider_response",
                        action_id=str(run_id),
                        usage=getattr(message, "usage_metadata", None),
                        model_name=metadata.get("model_name"),
                        finish_reason=metadata.get("finish_reason"),
                        tool_names=[c["name"] for c in getattr(message, "tool_calls", [])],
                    )

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.budget.after_model(str(run_id), None, error=True)
        self.budget.store.event(
            "provider_model_error", action_id=str(run_id), **model_error_details(error)
        )


def mock_usage(messages, output: str) -> dict:
    before = max(1, math.ceil(sum(len(str(m.content)) for m in messages) / 4))
    after = max(1, math.ceil(len(output) / 4))
    return {"input_tokens": before, "output_tokens": after, "total_tokens": before + after}
