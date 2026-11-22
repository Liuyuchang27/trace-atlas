"""Retry individual DeepSeek requests above the SDK so each attempt is metered."""

import asyncio
from asyncio import sleep as wait_before_retry
from uuid import uuid4

import httpx
from langchain.chat_models import init_chat_model as init_provider_model
from langchain_deepseek import ChatDeepSeek
from openai import APIConnectionError, APIStatusError

from .budget import BudgetCallback, BudgetExceeded, model_error_details

RETRY_DELAYS = (12, 16, 20)
CONNECT_TIMEOUT = 15
TRANSIENT_STATUS = (502, 503, 504)


def policy():
    return {
        "max_retries": len(RETRY_DELAYS),
        "delays_seconds": list(RETRY_DELAYS),
        "connect_timeout_seconds": CONNECT_TIMEOUT,
        "retry_http_status": list(TRANSIENT_STATUS),
        "retry_transport_errors": True,
        "usage": "each_attempt_metered_unknown_usage_reserved",
    }


def can_retry(error):
    return isinstance(error, APIConnectionError) or (
        isinstance(error, APIStatusError) and error.status_code in TRANSIENT_STATUS
    )


class RetryingDeepSeek(ChatDeepSeek):
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if payload.get("response_format") == {"type": "json_object"}:
            # JSON mode needs no SDK schema parsing. The parse endpoint raises on
            # length before callbacks see returned usage or our validators see the
            # finish reason. Keep the same wire body via the normal create endpoint.
            payload["extra_body"] = {
                **(payload.get("extra_body") or {}),
                "response_format": payload.pop("response_format"),
            }
        return payload

    async def agenerate(self, messages, stop=None, callbacks=None, *, run_id=None, **kwargs):
        handlers = callbacks if isinstance(callbacks, list) else getattr(callbacks, "handlers", ())
        budget = next((h.budget for h in handlers if isinstance(h, BudgetCallback)), None)
        generate = super().agenerate
        if budget is None or len(messages) != 1:
            # Do not replay successful members of a batch or make unmetered retries.
            return await generate(messages, stop, callbacks, run_id=run_id, **kwargs)

        retry_id = None
        phase = budget.state["phase"]
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                if retry_id:
                    await wait_before_retry(RETRY_DELAYS[attempt - 1])
                    budget.refresh_phase()
                    if budget.remaining_time() <= 0:
                        raise BudgetExceeded("deadline_reached")
                    if phase == "research" and budget.state["phase"] != phase:
                        raise BudgetExceeded("research_budget_exhausted")
                    budget.store.event("model_retry_started", retry_id=retry_id)
                result = await generate(messages, stop, callbacks, run_id=run_id, **kwargs)
            except asyncio.CancelledError:
                if retry_id:
                    budget.store.event(
                        "model_retry_finished", retry_id=retry_id, status="cancelled"
                    )
                raise
            except Exception as exc:
                details = model_error_details(exc)
                if retry_id:
                    budget.store.event(
                        "model_retry_finished", retry_id=retry_id, status="failed", **details
                    )
                if (
                    not can_retry(exc)
                    or attempt == len(RETRY_DELAYS)
                    or budget.remaining_time() <= RETRY_DELAYS[attempt]
                    or (
                        budget.config.max_model_calls is not None
                        and budget.state["model_calls"] >= budget.config.max_model_calls
                    )
                ):
                    raise
                run_id = uuid4()
                retry_id = str(run_id)
                budget.store.event(
                    "model_retry_scheduled",
                    retry_id=retry_id,
                    model=self.model_name,
                    phase=phase,
                    retry_number=attempt + 1,
                    max_retries=len(RETRY_DELAYS),
                    delay_seconds=RETRY_DELAYS[attempt],
                    connect_timeout_seconds=CONNECT_TIMEOUT,
                    **details,
                )
            else:
                if retry_id:
                    budget.store.event(
                        "model_retry_finished", retry_id=retry_id, status="succeeded"
                    )
                return result


def init_chat_model(model, **kwargs):
    if not model.startswith("deepseek:"):
        return init_provider_model(model, **kwargs)
    timeout = kwargs.get("timeout")
    if isinstance(timeout, int | float):
        kwargs["timeout"] = httpx.Timeout(timeout, connect=min(CONNECT_TIMEOUT, timeout))
    # Keep SDK retries disabled. The outer retry loop assigns a new callback run ID.
    return RetryingDeepSeek(model=model.split(":", 1)[1], **{**kwargs, "max_retries": 0})
