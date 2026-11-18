"""Lexical selection and explicit delivery, without changing saved source text."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

STOP = set(
    "a an the of to in on at for from with by and or as is are was were be been it its "
    "this that these those which who what when where how please find official website "
    "www http https com html pdf".split()
)

CORE_PROMPT = """
EVIDENCE-FIRST RESEARCH:
Investigate the user's main status question before optional history. Prefer a decisive original
record for the target status, then fill the announcement/regulatory timeline. Search titles and
your prior knowledge are leads, not proof. Record concise material findings with record_claim
as soon as useful original passages are available; record separate findings in one tool batch.
First reads select relevant exact passages from the page you choose. Use find_saved_text for a
specific missing phrase/qualifier, not repeated whole-page reads. Repeating a query may return
only already-delivered IDs; the selected originals remain in the saved evidence packet.
Distinguish FULL DOCUMENT SAVED, PASSAGE SHOWN, and CLAIM SUPPORTED. A short packet is a selection,
not the complete contents of disk. Do not claim that only navigation was saved if body passages
have been supplied. Use the packet's original text and its inventory; explain the precise missing
record or unsupported conclusion. Plan labels/dates remain tentative until original text supports
them. Preserve planned/observed distinctions. Claims are unverified, never certified facts.
SOURCE AND DATE GAPS:
Obtain a dated issuer/official record of the main event before optional announcement history.
A press release, corporate blog or filing can be primary: do not confine searches to one guessed
newsroom domain. Select include_domains yourself when useful. If a media article lacks the
original, follow its relevant links or search the exact title/event on the responsible body's
sites. If an official page refers to a separate/new proceeding, follow that page rather than
treating an older proceeding as the latest record. list_source_links shows leads from the full
saved page; open_url reads a link YOU select. Link labels and candidate dates are not proof.
Use target-nearest dated decisions: an earlier provisional finding does not answer a later
provisional decision question. Read the surrounding date/heading with the event paragraph.
Cite the original date heading too when the event paragraph does not itself state the date.
An article's URL date, page update, extraction clock or approval date is not a closing date.
Do not add a timezone absent from the original. Clearly distinguish direct statements from
inferences. If only an approval is collected, completion stays unknown, including in the headline
and timeline. Later completion evidence can establish its date; state separately what was known
by the requested earlier date. Do not interpret a missing closing record as non-completion.
Batch a few decisive searches/reads early. Before repeating a query, use pending candidate/link
leads or change the source path. Record each material finding once; claim recording is available
alongside research and must not replace acquiring the missing main-event record.
Omit transaction terms that do not answer the question. If a future production/release window is
material, keep it attached to eligible games/content, not the duration of ownership or rights.
"""


def core_policy():
    return {
        "version": "live-evidence-chain-v6",
        "gap_repair": "bounded-gap-repair-execution-v1",
        "online_questions": "question-evidence-state-v1.1",
        "model_request_timeout_seconds": 60,
        "transport_failure_finalization": "single_reserved_revision_if_sources_time_budget_allow",
        "completion_answers": "exact_quote_event_date_bindings_then_calendar_cutoff_inference",
        "selection": "idf_coverage_phrases_surface_penalty",
        "first_read_passages": 3,
        "focus_passages": 18,
        "context_bytes": 10000,
        "research_claim_turns": 0,
        "writing_claim_turns": 1,
        "source_links": "bounded_exact_saved_link_leads_agent_selected_followup",
        "candidate_memory": "pending_earlier_and_recent_candidates",
        "date_context": "adjacent_original_date_headings",
        "domain_filters": "agent_selected_optional_hostnames",
        "budget_precheck": "includes_actual_system_message_and_research_tool_schemas",
        "main_goal_search_turns": 1,
        "extra_model_budget": 0,
        "scope": "common live/visual layer; legacy local comparison unchanged",
    }


def tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    # URL targets are not original prose or proof of dates.
    text = re.sub(r"\]\(https?://[^)]+\)", "]", text)
    words = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text)
    result = []
    for word in words:
        if re.search(r"[\u3400-\u9fff]", word) and len(word) > 2:
            result.extend(word[i : i + 2] for i in range(len(word) - 1))
        elif word not in STOP:
            result.append(word)
    return result


def surface_penalty(text):
    """A presentation heuristic, not a truth or source-authority judgment."""
    lower = text.casefold()
    links = len(re.findall(r"\]\(https?://", text))
    visible = re.sub(r"https?://[^)\s]+", "", text)
    penalty = min(0.65, links * 24 / max(100, len(visible)))
    if len(re.findall(r"cookie|privacy|sign in|sign up|skip to|accept all|reject all", lower)) >= 2:
        penalty += 0.55
    if re.search(
        r"forward.looking statements|safe harbor|risk factors|actual results (?:could|may)", lower
    ):
        penalty += 0.28  # still retrievable for a query about the risks themselves
    if len(tokens(text)) < 8:
        penalty += 0.35
    elif re.match(r"^\s*#{1,6}\s", text) and "\n" not in text.strip():
        penalty += 0.35
    return min(0.9, penalty)


def rank_passages(passages, query):
    passages = list(passages)
    q = tokens(query)
    terms = set(q)
    bags = [Counter(tokens(p.text)) for p in passages]
    df = {t: sum(t in bag for bag in bags) for t in terms}
    weights = {t: 1 + math.log((len(bags) + 1) / (df[t] + 1)) for t in terms}
    denominator = sum(weights.values()) or 1
    phrase = " ".join(q)
    pairs = {tuple(q[i : i + 2]) for i in range(len(q) - 1)}
    triples = {tuple(q[i : i + 3]) for i in range(len(q) - 2)}
    ranked = []
    for p, bag in zip(passages, bags, strict=True):
        matched = terms & bag.keys()
        sequence = tokens(p.text)
        normalized = " ".join(sequence)
        coverage = sum(weights[t] for t in matched) / denominator
        adjacent = sum(" ".join(pair) in normalized for pair in pairs)
        adjacent += 2 * sum(" ".join(triple) in normalized for triple in triples)
        score = 8 * coverage + sum(weights[t] for t in matched) / (1 + len(sequence) / 250)
        score += min(adjacent, 20) * 0.65
        if phrase and phrase in normalized:
            score += 15
        penalty = surface_penalty(p.text)
        score *= 1 - penalty
        if not matched:
            score = 0.05 * (1 - penalty)
        ranked.append(
            {
                "passage": p,
                "score": round(score, 6),
                "surface_penalty": penalty,
                "matched_terms": sorted(matched),
            }
        )
    ranked.sort(key=lambda row: (-row["score"], row["passage"].start))
    return ranked


def expose_passages(evidence, budget, passages, *, method, query="", scores=None):
    """Record exact text exposure separately from document storage and claim support."""
    from .harness import remember_passages

    passages = list(passages)
    remember_passages(budget, [p.passage_id for p in passages])
    state = getattr(evidence, "question_state", None)
    if state:
        state.receive(passages)
    focus = budget.state.setdefault("evidence_focus", {})
    if scores is None:
        scores = {r["passage"].passage_id: r for r in rank_passages(passages, query)}
    for p in passages:
        row = scores.get(p.passage_id, {})
        penalty = row.get("surface_penalty", surface_penalty(p.text))
        if method in {"ranked_read", "find_saved_text"} and penalty < 0.7:
            previous = focus.get(p.passage_id, {})
            focus[p.passage_id] = {
                "score": max(previous.get("score", 0), row.get("score", 1)),
                "method": method,
                "query": query[:160],
                "version_id": p.version_id,
            }
    keep = sorted(focus, key=lambda pid: -focus[pid]["score"])[: core_policy()["focus_passages"]]
    budget.state["evidence_focus"] = {pid: focus[pid] for pid in keep}
    budget.save()
    evidence.store.event(
        "evidence_delivered",
        method=method,
        query=query[:160],
        passages=[{"passage_id": p.passage_id, "start": p.start, "end": p.end} for p in passages],
    )


def claim_needed(evidence, budget):
    focused = set(budget.state.get("evidence_focus", {}))
    return (
        budget.state["phase"] == "writing"
        and bool(focused)
        and not evidence.claims
        and budget.state.get("core_claim_turns", 0) < core_policy()["writing_claim_turns"]
    )


def unstarted_main_goals(evidence, budget):
    """Model-declared open needs schedule an action; their truth is never certified here."""
    from .storage import read_json

    path = evidence.store.root / "public_plan.json"
    if (
        budget.state["phase"] != "research"
        or not evidence.documents
        or budget.state.get("main_goal_search_turns", 0) >= core_policy()["main_goal_search_turns"]
        or not path.exists()
    ):
        return []
    return [
        {"step_id": step["step_id"], "title": step["title"]}
        for step in read_json(path).values()
        if step.get("answer_role") == "main_status" and step["status"] == "planned"
    ]


def with_date_context(passages, selected, *, publication_heading=False):
    """Include exact adjacent date/heading paragraphs, never manufacture metadata dates."""
    passages = sorted(passages, key=lambda p: p.start)
    result = list(selected)
    date = re.compile(
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\d{4}年\d{1,2}月\d{1,2}日|"
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b|"
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\b",
        re.I,
    )
    for p in selected:
        index = passages.index(p)
        if publication_heading:
            # A short labelled page heading can be separated from the body by share
            # widgets. Scan across those paragraphs, but never borrow an arbitrary
            # historic date or a date from another saved document.
            for previous in reversed(passages[:index]):
                if p.start - previous.end > 1800:
                    break
                if (
                    len(previous.text) <= 220
                    and date.search(previous.text)
                    and re.search(
                        r"发布时间|发布日期|发表时间|发布于|\bpublished(?:\s+(?:on|at))?\b",
                        previous.text,
                        re.I,
                    )
                ):
                    result.append(previous)
                    break
        for previous in reversed(passages[max(0, index - 3) : index]):
            if p.start - previous.end > 500 or len(previous.text) > 220:
                break
            if date.search(previous.text):
                result.append(previous)
                break
    return list({p.passage_id: p for p in result}.values())


def with_prose_context(passages, selected, max_bytes=3000):
    """Bounded verbatim neighbours preserve qualifications lost by lexical ranking.

    This selects prose, not truth or independence. No new retrieval or invented text.
    Existing ranked/date selections are never removed by the supplemental cap.
    """
    ordered = sorted(passages, key=lambda p: p.start)
    result = list(selected)
    seen = {p.passage_id for p in result}
    anchors = [i for i, p in enumerate(ordered) if p.passage_id in seen]
    used = sum(len(p.text.encode()) for p in result)
    for distance in range(1, 4):
        for i in anchors:
            for j in [i + distance, i - distance]:
                if not 0 <= j < len(ordered):
                    continue
                p = ordered[j]
                if p.passage_id in seen or surface_penalty(p.text) >= 0.5:
                    continue
                size = len(p.text.encode())
                if used + size <= max_bytes:
                    result.append(p)
                    seen.add(p.passage_id)
                    used += size
    return result
