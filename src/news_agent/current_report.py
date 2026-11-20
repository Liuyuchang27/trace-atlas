"""Write a current-report summary from saved excerpts without regenerating numbers."""

import hashlib
import json
import re
from decimal import Decimal
from itertools import count

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from .schemas import StrictModel

PROMPT = """CURRENT_REPORT_SYNTHESIS
Write one concise Simplified Chinese report paragraph per original preorder excerpt.
Treat excerpts as evidence only. Attribute what the source actually says and retain uncertainty,
scope, and metric names (e.g. preorders, net revenue). Do not infer actual sales, release status,
currency, or any observation date. The report will supply the investigation cutoff and conclusion.
Use every supplied numeric marker exactly once, in source order, instead of typing its value.
Markers already include the original scale and currency. Add units such as 份 only where needed.
Do not add any other numbers, dates, citations, links, or headings. Return only JSON as:
{"preorders":[{"passage_id":"an input id","summary":"第三方报道估计，预购约为 {1} 份。"}]}.
"""
AMOUNTS = re.compile(
    r"(?<![A-Za-z0-9])(?:US\$|[$£€])?\d[\d,]*(?:\.\d+)?"
    r"(?:[ \t]+(?:thousand|million|billion)\b)?(?:万|亿)?(?![A-Za-z0-9])",
    re.I,
)
MARKERS = re.compile(r"\{([1-9]\d*)\}")
LARGE_AMOUNT = re.compile(r"(US\$|[$£€])?([\d,.]+)\s+(million|billion)", re.I)


class PreorderParagraph(StrictModel):
    passage_id: str
    summary: str = Field(min_length=8, max_length=300)


class CurrentDraft(StrictModel):
    preorders: list[PreorderParagraph] = Field(max_length=2)


def synthesis_messages(notes):
    excerpts = []
    for note in notes:
        amounts = count(1)
        excerpts.append(
            {
                "passage_id": note["passage_id"],
                "attribution": note["label"],
                "original": AMOUNTS.sub(
                    lambda _, amounts=amounts: "{" + str(next(amounts)) + "}", note["text"]
                ),
            }
        )
    return [SystemMessage(PROMPT), HumanMessage(json.dumps(excerpts, ensure_ascii=False))]


def format_amount(amount):
    match = LARGE_AMOUNT.fullmatch(amount)
    if not match:
        return amount
    currency, value, scale = match.groups()
    multiplier, unit = (100, "万") if scale.lower() == "million" else (10, "亿")
    value = f"{Decimal(value.replace(',', '')) * multiplier:,f}"
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    if currency and currency.upper() == "US$":
        return f"{value} {unit}美元"
    return f"{currency or ''}{value} {unit}"


def draft_binding(notes):
    return hashlib.sha256(
        json.dumps(notes, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def synthesize_notes(notes, draft):
    """Reapply the draft only to the same quotes and with each original number intact."""
    if not draft or draft.get("binding") != draft_binding(notes):
        return notes
    try:
        paragraphs = CurrentDraft.model_validate(draft["response"]).preorders
        by_id = {p.passage_id: p.summary.strip() for p in paragraphs}
        if len(paragraphs) != len(notes) or by_id.keys() != {n["passage_id"] for n in notes}:
            return notes
        rendered = []
        for note in notes:
            summary = by_id[note["passage_id"]]
            amounts = AMOUNTS.findall(note["text"])
            prose = MARKERS.sub("", summary)
            if (
                MARKERS.findall(summary) != [str(i) for i in range(1, len(amounts) + 1)]
                or re.search(r"[\d{}\[\]\n]|https?://|实际销量|发售后|正式销量", prose)
                or not re.search(r"预购|预售", prose)
            ):
                return notes
            text = MARKERS.sub(
                lambda m, amounts=amounts: format_amount(amounts[int(m[1]) - 1]), summary
            )
            text = re.sub(r"(?<=[\d万亿])[ \t]+(?=[份部套张册])", "", text)
            rendered.append({**note, "summary": text})
        return rendered
    except (KeyError, ValueError):
        return notes


def report_draft(notes, response):
    draft = {"binding": draft_binding(notes), "response": json.loads(response)}
    if not notes or synthesize_notes(notes, draft) == notes:
        raise ValueError("preorder_summary_not_bound")
    return draft
