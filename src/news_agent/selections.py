"""Select bounded verbatim excerpts; model responses never supply quote text."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

from pydantic import Field

from .schemas import StrictModel


class TemporalSelection(StrictModel):
    stage: Literal["announcement", "approval", "completion", "other", "unknown"]
    kind: Literal["observed", "planned", "unknown"]
    event_date: date | None = None
    verdict: Literal["supported", "insufficient", "contradicted"]
    passage_id: str
    excerpt_id: str | None
    reason: str = Field(min_length=1, max_length=500)


class SupportSelectionLink(StrictModel):
    passage_id: str
    relation: Literal["supports", "contradicts", "related", "insufficient"]
    excerpt_id: str | None
    reason: str = Field(min_length=1, max_length=400)


class SupportSelection(StrictModel):
    links: list[SupportSelectionLink] = Field(min_length=1, max_length=6)


class OriginSelectionQuote(StrictModel):
    version_id: str
    excerpt_id: str


class OriginSelection(StrictModel):
    relation: Literal["same_origin", "cites", "distinct_origin", "unknown"]
    confidence: Literal["confirmed", "suspected", "unknown"]
    evidence: list[OriginSelectionQuote] = Field(max_length=4)
    needs_independent_record: bool
    reason: str = Field(min_length=1, max_length=500)


SELECTION_SCHEMAS = {"T": TemporalSelection, "S": SupportSelection, "ORIGIN": OriginSelection}


def excerpts(source_id, text, width=420, overlap=120):
    """Overlapping original character spans, with no trimming or normalization."""
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + width)
        identity = f"{source_id}:{start}:{end}:{text[start:end]}"
        result.append(
            {
                "excerpt_id": "x-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )
        if end == len(text):
            break
        start = end - overlap
    return result


def review_payload(kind, payload):
    key, id_key = ("documents", "version_id") if kind == "ORIGIN" else ("passages", "passage_id")
    catalog, rows = {}, []
    for source in payload[key]:
        choices = excerpts(source[id_key], source["text"])
        rows.append({**{k: v for k, v in source.items() if k != "text"}, "excerpts": choices})
        for choice in choices:
            catalog[choice["excerpt_id"]] = {
                **choice,
                "source_id": source[id_key],
                "original": source["text"],
            }
    return {**payload, key: rows}, catalog


def resolve_selection(kind, selected, catalog):
    value = selected.model_dump(mode="json")

    def quote(item, id_key):
        excerpt_id = item.pop("excerpt_id")
        if excerpt_id is None:
            item["quote"] = ""
            return
        entry = catalog.get(excerpt_id)
        if entry is None or entry["source_id"] != item[id_key]:
            raise ValueError("excerpt_not_in_supplied_source")
        text = entry["original"][entry["start"] : entry["end"]]
        if text != entry["text"] or not text:
            raise ValueError("excerpt_offsets_do_not_match_original")
        item.update(quote=text, quote_span={"start": entry["start"], "end": entry["end"]})

    if kind == "T":
        quote(value, "passage_id")
    elif kind == "S":
        for link in value["links"]:
            quote(link, "passage_id")
    else:
        for item in value["evidence"]:
            quote(item, "version_id")
    return value
