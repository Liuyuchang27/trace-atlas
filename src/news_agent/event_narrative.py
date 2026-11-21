"""Render event reports with section-level evidence and one source entry per document."""

import re

from pydantic import Field

from .schemas import StrictModel


class Paragraph(StrictModel):
    text: str = Field(min_length=1)
    findings: list[int] = Field(description="Numbers of accepted findings supporting this text.")


class Narrative(StrictModel):
    conclusion: Paragraph
    cause: Paragraph
    turning_point: Paragraph
    outcome: Paragraph


def render_narrative(draft, facts, evidence, chinese):
    labels = (
        ["结论", "起因", "转折", "实际结果"]
        if chinese
        else ["Conclusion", "Background", "Turning point", "Outcome"]
    )
    sources = {}
    sections = []
    bindings = {}
    for label, para in zip(
        labels, [draft.conclusion, draft.cause, draft.turning_point, draft.outcome], strict=True
    ):
        if any(not 1 <= n <= len(facts) for n in para.findings):
            raise ValueError("Choose only supplied finding numbers")
        if re.search(r"\[p-|https?://", para.text):
            raise ValueError("References belong in findings, not in paragraph text")
        sections.append(f"## {label}\n\n{para.text.strip()}")
        bindings[label] = []
        for number in dict.fromkeys(para.findings):
            for ref in facts[number - 1]["originals"]:
                pid = ref["passage_id"]
                doc = evidence.documents[evidence.passages[pid].version_id]
                entry = sources.setdefault(doc.url or doc.version_id, {"pid": pid, "sections": []})
                if label not in entry["sections"]:
                    entry["sections"].append(label)
                if pid not in bindings[label]:
                    bindings[label].append(pid)
    if facts and (not draft.conclusion.findings and not draft.outcome.findings):
        raise ValueError("Ground the answer or observed outcome in an accepted finding")
    if sources:
        lines = ["## " + ("原文依据" if chinese else "Sources")]
        for source in sources.values():
            lines.append("、".join(source["sections"]) + f" [{source['pid']}]")
        sections.append("\n\n".join(lines))
    return "\n\n".join(sections), bindings
