"""Explicit stage profiles; one agent, one ledger, no hidden fallback or extra calls."""

from __future__ import annotations

from .model_retry import init_chat_model
from .model_retry import policy as retry_policy
from .resource_control import execution_limits

PROFILE = "staged-deepseek-v1"
FLASH = "deepseek:deepseek-flash"
PRO = "deepseek:deepseek-v4-pro"
REVIEW_REASONING_EFFORT = "high"
STAGES = {
    "scope_planning": (PRO, False),
    "gap_planning": (PRO, False),
    "retrieval_selection": (FLASH, False),
    "evidence_assessment": (PRO, False),
    "evidence_repair": (PRO, False),
    "report_draft": (FLASH, False),
    "report_review": (PRO, True),
    "temporal_review": (PRO, True),
}


OUTPUT_TARGETS = {
    "scope_planning": 10000,
    "gap_planning": 10000,
    "retrieval_selection": 6000,
    "evidence_assessment": 12000,
    "evidence_repair": 12000,
    "report_draft": 16000,
    "report_review": 16000,
    "temporal_review": 12000,
}


def output_limit(stage, config):
    return min(config.max_output_tokens, OUTPUT_TARGETS[stage])


def policy():
    return {
        "version": PROFILE,
        "stages": {
            stage: {
                "model": model,
                "thinking": "enabled" if thinking else "disabled",
                "reasoning_effort": REVIEW_REASONING_EFFORT if thinking else None,
            }
            for stage, (model, thinking) in STAGES.items()
        },
        "budget": "shared_original_total_and_output_cap",
        "output_targets": OUTPUT_TARGETS,
        "output_rule": "min(stage_target, budget.max_output_tokens)",
        "retries": retry_policy(),
        "fallback": "none",
        "thinking_scope": "isolated_tool_free_temporal_and_final_review",
        "reasoning_visibility": "never_export_private_reasoning",
    }


def stage_for(tools, writing=False):
    names = {t.name if hasattr(t, "name") else t.get("name") for t in tools}
    if "resolve_event_write" in names:
        return "evidence_repair"
    if "begin_investigation" in names:
        return "scope_planning"
    if "read_current_sources" in names:
        return "retrieval_selection"
    if names & {
        "append_event_finding",
        "create_event_node",
        "append_undated_finding",
        "record_event_finding",
        "record_event_relation",
        "review_node_sources",
        "record_current_result",
    }:
        return "evidence_assessment"
    if writing or not names:
        return "report_draft"
    if names & {"search_gap", "replan_gap", "search_current_sources"}:
        return "gap_planning"
    return "retrieval_selection"


class StageRouter:
    def __init__(self, budget):
        self.budget = budget
        self.models = {}

    def select(self, stage):
        model_name, thinking = STAGES[stage]
        key = (model_name, thinking, output_limit(stage, self.budget.config))
        if key not in self.models:
            options = {"extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}}}
            if thinking:
                options["reasoning_effort"] = REVIEW_REASONING_EFFORT
            self.models[key] = init_chat_model(
                model_name,
                temperature=0,
                max_tokens=output_limit(stage, self.budget.config),
                timeout=max(
                    1,
                    min(
                        execution_limits(self.budget.config)["model_timeout_seconds"],
                        self.budget.remaining_time(),
                    ),
                ),
                max_retries=0,
                **options,
            )
        self.budget.store.event(
            "model_route",
            profile=PROFILE,
            stage=stage,
            max_output_tokens=output_limit(stage, self.budget.config),
            **policy()["stages"][stage],
        )
        return self.models[key]
