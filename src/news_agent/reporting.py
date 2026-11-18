"""Reference validation is mechanical; it does not certify factual support."""

from __future__ import annotations

import re

from .storage import EvidenceStore


def partial_report(evidence: EvidenceStore, reason: str) -> str:
    lines = ["# Partial research report", "", f"Stop reason: {reason}", ""]
    lines += ["Only unverified recorded claims are shown; no completion is inferred.", ""]
    for claim in evidence.claims:
        lines.append("- " + claim.text + " " + " ".join(f"[{p}]" for p in claim.passage_ids))
    if not evidence.claims:
        lines.append("No claims were recorded before stopping.")
    return "\n".join(lines)


def validate_and_append_sources(text: str, evidence: EvidenceStore) -> tuple[str, list[str]]:
    references = set(re.findall(r"\[(p-[a-f0-9]{16})\]", text))
    invalid = sorted(references - evidence.passages.keys())
    for pid in invalid:
        text = text.replace(f"[{pid}]", "[INVALID CITATION REMOVED]")
    sources = []
    for pid in sorted(references & evidence.passages.keys()):
        passage = evidence.passages[pid]
        doc = evidence.documents[passage.version_id]
        sources.append(
            f"- [{pid}] {doc.title} — {doc.url} "
            f"(version: {doc.version_id}, offsets: {passage.start}:{passage.end})"
        )
    text += "\n\n## Evidence index\n\n" + ("\n".join(sources) or "No valid passage references.")
    return text + "\n", invalid
