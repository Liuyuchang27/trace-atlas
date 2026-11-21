"""The common task/evidence contract. No gold answers belong in these models."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def aware(value: datetime | None) -> datetime | None:
    if value is not None and value.utcoffset() is None:
        raise ValueError("timestamps must include an explicit timezone")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchTask(StrictModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    event: str = Field(min_length=1)
    question: str = Field(min_length=1)
    mode: Literal["as_of", "retrospective"]
    cutoff: datetime | None = None
    synthetic: bool = False

    _aware = field_validator("cutoff")(aware)

    @model_validator(mode="after")
    def require_cutoff(self):
        if self.mode == "as_of" and self.cutoff is None:
            raise ValueError("as_of tasks require a cutoff")
        return self


class DocumentVersion(StrictModel):
    version_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    title: str
    url: str
    content: str = Field(min_length=1)
    published_at: datetime | None = None
    available_at: datetime | None = None
    availability_proof: str | None = None
    retrieved_at: datetime | None = None

    _aware = field_validator("published_at", "available_at", "retrieved_at")(aware)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()


class Passage(StrictModel):
    passage_id: str
    version_id: str
    start: int
    end: int
    text: str


class Claim(StrictModel):
    claim_id: str
    text: str = Field(min_length=1)
    passage_ids: list[str] = Field(min_length=1)
    event_time: str | None = None
    stage: str | None = None
    verification: Literal["unverified"] = "unverified"


class BudgetConfig(StrictModel):
    max_model_calls: int | None = Field(default=24, ge=1)
    max_tool_calls: int | None = Field(default=32, ge=1)
    max_tokens: int | None = Field(default=80000, ge=1)
    deadline_seconds: float | None = Field(default=600, gt=0)
    reserve_model_calls: int = Field(default=6, ge=0)
    reserve_tool_calls: int = Field(default=8, ge=0)
    reserve_tokens: int = Field(default=20000, ge=0)
    reserve_seconds: float = Field(default=120, ge=0)
    max_output_tokens: int = Field(default=1500, ge=1)
    planning_output_tokens: int | None = Field(default=None, ge=1, exclude_if=lambda v: v is None)
    review_output_tokens: int | None = Field(default=None, ge=1, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def valid_reserves(self):
        for total, reserve in [
            (self.max_model_calls, self.reserve_model_calls),
            (self.max_tool_calls, self.reserve_tool_calls),
            (self.max_tokens, self.reserve_tokens),
            (self.deadline_seconds, self.reserve_seconds),
        ]:
            if total is not None and reserve >= total:
                raise ValueError("each reserve must be smaller than its total")
        return self
