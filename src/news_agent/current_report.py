"""Write a current-report summary from saved excerpts without regenerating numbers."""

import hashlib
import json
import re
from decimal import Decimal
from itertools import count
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from .schemas import StrictModel

ANSWER_FIRST = """Start from the user's original question. Identify the entity, requested fact or
measure, time and scope. Lead with the best directly supported answer. For a quantitative question
give the value, unit, reporting period and attribution when known. If the requested figure is
unavailable, say exactly what could be established and where the check ended. Distinguish verified
results from estimates, forecasts and prior periods. Include a related finding only if it materially
helps answer THIS question and its different scope is explicit. For each sentence ask which part of
the user's question it answers. Omit statements about a different entity, measure or period unless
the difference is essential to interpret the answer. Stop each sentence once it answers the question.
Omit rankings, records and management praise unless the user actually asked for a comparison. A company's outlook cannot stand in for a
product result. An unsuccessful search supports only a statement about the originals inspected.
Citations demonstrate support; the existence of an excerpt is never a reason to include it."""

PROMPT = (
    """CURRENT_REPORT_SYNTHESIS
"""
    + ANSWER_FIRST
    + """
Write in the language of the question. Answer ONLY the original question in the first sentence.
Compose the answer using the assessed result and eligible answer passage IDs. If unconfirmed,
clearly limit the answer to the checked sources. Do not introduce numbers outside the assessed result.
The reader will see launch proof and its date separately. For secondary excerpts, retain only directly
relevant quantified observations; omit the rest. Preserve all numeric markers in each retained
excerpt once in order, without adding new values. Name the reporting source and distinguish estimates
from achieved units. Do not give a publication date as the measurement cutoff.
If metric_evidence is present, compose findings from those actual original statements. First pick
the closest OFFICIAL observation in the requested unit; retain its precise time basis and clearly
identify a presale measure. Include the officially measured period in the sentence (for example 发售前 or 首日). Where a footnote supplies the measurement end date, use its exact meaning and put its date in period rather than typing digits in summary. Then include official money amounts in their own labeled measure and
use third-party estimates only as separately attributed comparisons. Never divide revenue by a price.
Skip other products, forecasts, adjectives, reviews and ratings. Use {1}, {2}, etc. from the SAME
excerpt for the amounts being reported. Omit incidental numbers and do not type numerals in findings.
Each summary is one plain Chinese sentence. Preserve the currency with the amount marker; do not
repeat its currency or scale. Cite only the given passage_id. Distinguish sell-in from sell-through.
For this mode, answer is a short statement about what remains unknown or verified as of the check;
the first accepted finding will lead the report. For an unresolved result use a bounded statement
such as 本次查阅的原文未给出… . Do not state that the issuer has never disclosed it. Never say only
third-party data exists. If quantified observations exist, include at least one supported finding. Do not repeat figures or say there is no evidence
when a narrower period IS documented. Use preorders=[] if a finding already covers an estimate.
Return only JSON {"findings":[{"scope":"direct|closest_official|external_estimate","passage_id":"...",
"summary":"...","period":"exact substring from basis or null"}],"answer":"...", "answer_passages":["..."], "preorders":[{"passage_id":"...",
"summary":"..."}]}. Empty citation and secondary lists are valid if there is no usable evidence.
"""
)
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


class ReportPoint(StrictModel):
    period: str | None = Field(default=None, max_length=140)
    passage_id: str
    scope: Literal["direct", "closest_official", "external_estimate"]
    summary: str = Field(min_length=3, max_length=350)


class CurrentDraft(StrictModel):
    findings: list[ReportPoint] = Field(default_factory=list, max_length=5)
    answer: str | None = Field(default=None, max_length=450)
    answer_passages: list[str] = Field(default_factory=list, max_length=8)
    preorders: list[PreorderParagraph] = Field(max_length=2)


def marked(text):
    sequence = count(1)
    return AMOUNTS.sub(lambda _: "{" + str(next(sequence)) + "}", text)


def bound_points(state, draft):
    if not draft or draft.get("answer_binding") != draft_binding(answer_context(state)):
        return []
    evidence = {r["passage_id"]: r for r in state.get("metric_evidence", [])}
    points = []
    parsed = CurrentDraft.model_validate(draft["response"])
    claims = list(parsed.findings)
    for paragraph in parsed.preorders:
        original = evidence.get(paragraph.passage_id)
        if (
            original
            and not original["official"]
            and re.search(r"pre.?order|预购|预售", original["text"], re.I)
        ):
            claims.append(ReportPoint(**paragraph.model_dump(), scope="external_estimate"))
    for claim in claims:
        r = evidence.get(claim.passage_id)
        if not r or r["official"] != (claim.scope != "external_estimate"):
            continue
        nums = AMOUNTS.findall(r["text"])
        ids = [int(i) for i in MARKERS.findall(claim.summary)]
        plain = MARKERS.sub("", claim.summary)
        raw_prose = plain
        for name, serial in re.findall(r"([A-Za-z]+)\s*(\d+)", state.get("original_question", "")):
            raw_prose = re.sub(rf"{re.escape(name)}\s*{serial}(?!\d)", "", raw_prose, flags=re.I)
        if not ids or any(i < 1 or i > len(nums) for i in ids) or re.search(r"\d|[{}]", raw_prose):
            continue
        if re.search(r"pre.release|pre.?order", r["text"], re.I) and not re.search(
            r"预售|预购|发售前|发行前|上市前", plain
        ):
            continue
        if re.search(r"pre.release|pre.?order", r["text"], re.I) and claim.scope == "direct":
            claim = claim.model_copy(update={"scope": "closest_official"})
        if any(re.search(r"[$£€]", nums[i - 1]) for i in ids) and not re.search(
            r"销售额|预售额|预购额|零售额|收入|金额|零售销售|revenue|sales revenue", plain, re.I
        ):
            continue
        if (
            r["official"]
            and re.search(r"unit|copies", r["text"], re.I)
            and not re.search(r"份|套|销量|copies|units", plain, re.I)
        ):
            continue
        summary = MARKERS.sub(
            lambda m, nums=nums: format_amount(nums[int(m[1]) - 1]), claim.summary
        )
        summary = re.sub(r"(?<=[万亿])\s+(?=[份套部])", "", summary)
        if any(point["summary"] == summary for point in points):
            continue
        if claim.period and any(claim.period in b["text"] for b in r["basis"]):
            summary += f"（统计时点原文：{claim.period}）"
        points.append(
            {
                **r,
                "scope": claim.scope,
                "summary": summary,
                "citations": [r["passage_id"], *[b["id"] for b in r["basis"]]],
            }
        )
    return points


def answer_context(state):
    from .current_check import result_view

    result = result_view(state)
    return {
        "question": state.get("original_question", ""),
        "cutoff": result["cutoff_label"],
        "outcome": result["outcome"],
        "assessed_answer": result["text"],
        "record_on": result["record_on"],
        "data_through": result.get("data_through"),
        "eligible_originals": [
            {"id": r["passage_id"], "quote": r["quote"]} for r in result.get("references", [])
        ],
        "release_record": (result.get("release") or {}).get("day"),
        "sales_question": bool(state.get("sales_question")),
        "metric_evidence": [
            {**r, "text": marked(r["text"])} for r in state.get("metric_evidence", [])
        ],
        "official_sources_read": {
            role: [x["url"] for x in route.get("originals", [])]
            for role, route in state.get("official_recovery", {}).get("routes", {}).items()
        },
    }


def synthesis_messages(notes, state=None):
    excerpts = []
    for note in notes:
        amounts = count(1)
        excerpts.append(
            {
                "passage_id": note["passage_id"],
                "attribution": note["label"],
                "source": note.get("source", ""),
                "original": AMOUNTS.sub(
                    lambda _, amounts=amounts: "{" + str(next(amounts)) + "}", note["text"]
                ),
            }
        )
    return [
        SystemMessage(PROMPT),
        HumanMessage(
            json.dumps(
                {**(answer_context(state) if state else {}), "related_excerpts": excerpts},
                ensure_ascii=False,
            )
        ),
    ]


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
        if (
            len(by_id) != len(paragraphs)
            or not by_id.keys() <= {n["passage_id"] for n in notes}
            or (not draft.get("answer_binding") and len(by_id) != len(notes))
        ):
            return notes
        rendered = []
        for note in notes:
            if note["passage_id"] not in by_id:
                continue
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


def answer_lead(state, draft):
    if not draft or draft.get("answer_binding") != draft_binding(answer_context(state)):
        return None
    context = answer_context(state)
    obj = CurrentDraft.model_validate(draft["response"])
    answer = (obj.answer or "").strip()
    if not answer or set(obj.answer_passages) - {r["id"] for r in context["eligible_originals"]}:
        return None
    known = " ".join(
        str(context[k] or "")
        for k in ("question", "assessed_answer", "record_on", "data_through", "cutoff")
    )
    if set(re.findall(r"\d+(?:[.,]\d+)*", answer)) - set(re.findall(r"\d+(?:[.,]\d+)*", known)):
        return None
    if context["outcome"] != "confirmed" and re.search(
        r"(?:均|从未|迄今|仍然|尚未|未曾)(?:正式|公开)?(?:公布|披露)|(?:可查到的)?仅是第三方|只有第三方",
        answer,
    ):
        return None
    if context["sales_question"]:
        if not re.search(r"销量|售出|销售量|sold|sales|出货", answer, re.I):
            return None
        if context["outcome"] != "confirmed" and not re.search(
            r"未|无[法从]|没有|不能|未能|could not|did not|not (?:found|verify|confirm)",
            answer,
            re.I,
        ):
            return None
    if context["outcome"] == "confirmed" and (
        not obj.answer_passages or (context["sales_question"] and not re.search(r"\d", answer))
    ):
        return None
    return {"text": answer, "passages": obj.answer_passages}


def report_draft(notes, response, state=None):
    draft = {"binding": draft_binding(notes), "response": json.loads(response)}
    if state is not None:
        draft["answer_binding"] = draft_binding(answer_context(state))
        CurrentDraft.model_validate(draft["response"])
        if (
            bound_points(state, draft)
            or answer_lead(state, draft)
            or (notes and synthesize_notes(notes, draft) != notes)
        ):
            return draft
        raise ValueError("answer_not_bound")
    if not notes or synthesize_notes(notes, draft) == notes:
        raise ValueError("preorder_summary_not_bound")
    return draft
