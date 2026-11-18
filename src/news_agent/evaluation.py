"""Mechanical checks plus explicit, auditable reviewer annotations, never an LLM judge."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

from .harness import file_hash
from .storage import read_json, write_json

REFERENCE = re.compile(r"\[(p-[a-f0-9]{16})\]")


def ratio(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_annotation(report: str, passages: set[str], reference: dict, annotation: dict) -> dict:
    if not annotation.get("review_complete"):
        raise ValueError("semantic scoring requires an explicitly completed review")
    truth_counts = {label: 0 for label in ["correct", "incorrect", "uncertain"]}
    pair_counts = {label: 0 for label in ["supported", "unsupported", "contradicted", "uncertain"]}
    covered = 0
    claim_ids = set()
    error_counts: dict[str, int] = {}
    for claim in annotation["claims"]:
        if claim["claim_id"] in claim_ids:
            raise ValueError("duplicate annotation claim ID")
        claim_ids.add(claim["claim_id"])
        if not claim["report_spans"]:
            raise ValueError("claim annotation needs a report span")
        cited_in_spans = set()
        for start, end in claim["report_spans"]:
            if not 0 <= start < end <= len(report):
                raise ValueError("invalid report annotation offsets")
            cited_in_spans.update(REFERENCE.findall(report[start:end]))
        truth_counts[claim["truth"]] += 1
        supported = False
        seen_pairs = set()
        for pair in claim["citations"]:
            pid = pair["passage_id"]
            if pid not in cited_in_spans or pid in seen_pairs:
                raise ValueError("citation annotation absent from report spans or duplicated")
            seen_pairs.add(pid)
            verdict = pair["verdict"]
            if verdict == "supported" and pid not in passages:
                raise ValueError("a nonexistent passage cannot support a claim")
            pair_counts[verdict] += 1
            supported |= verdict == "supported"
        covered += supported
        for category in set(claim.get("errors", [])):
            error_counts[category] = error_counts.get(category, 0) + 1
    expected = {q["question_id"] for q in reference["questions"]}
    scored = {q["question_id"] for q in annotation["question_scores"]}
    if expected != scored or len(scored) != len(annotation["question_scores"]):
        raise ValueError("reference question annotations are missing, extra or duplicated")
    scores = [q["score"] for q in annotation["question_scores"]]
    if any(s not in [0, 0.5, 1] for s in scores):
        raise ValueError("question scores must be 0, 0.5 or 1")
    if not claim_ids and any(scores):
        raise ValueError("an output with no assessed factual claims cannot receive answer credit")
    return {
        "reviewed_claims": len(claim_ids),
        "truth_counts": truth_counts,
        "citation_pair_counts": pair_counts,
        "claims_with_supporting_citation": covered,
        "reference_question_count": len(expected),
        "reference_question_score": sum(scores),
        "factual_correctness": ratio(
            truth_counts["correct"], truth_counts["correct"] + truth_counts["incorrect"]
        ),
        "citation_support": ratio(pair_counts["supported"], sum(pair_counts.values())),
        "citation_coverage": ratio(covered, len(claim_ids)),
        "answer_coverage": ratio(sum(scores), len(expected)),
        "errors": error_counts,
    }


def evaluate_pilot(pilot_path: Path, gold_path: Path, annotation_path: Path, output: Path) -> dict:
    pilot, gold, annotations = [read_json(p) for p in [pilot_path, gold_path, annotation_path]]
    if pilot["split"] != "development" or gold["split"] != "development":
        raise ValueError("this evaluator is for the development pilot only")
    if annotations["gold_sha256"] != file_hash(gold_path):
        raise ValueError("scoring reference changed after annotation")
    if annotations["pilot_sha256"] != file_hash(pilot_path):
        raise ValueError("pilot manifest changed after annotation")
    rows = []
    for case in pilot["cases"]:
        task_id, root = case["task_id"], Path(case["run_dir"])
        run, evidence, task = [
            read_json(root / p) for p in ["run.json", "evidence.json", "task.json"]
        ]
        annotation = annotations["cases"][task_id]
        if annotation["report_sha256"] != file_hash(root / "report.md"):
            raise ValueError("report changed after annotation")
        report = (root / "report.md").read_text().split("\n\n## Evidence index", 1)[0]
        refs = set(REFERENCE.findall(report))
        passages = set(evidence["passages"])
        semantic = score_annotation(report, passages, gold["cases"][task_id], annotation)
        cutoff = datetime.fromisoformat(task["cutoff"])
        documents = read_json(root / "input_corpus.json")
        boundary_errors = [
            d["version_id"]
            for d in documents
            if not d.get("available_at")
            or not d.get("availability_proof")
            or datetime.fromisoformat(d["available_at"]) > cutoff
        ]
        canary = gold["leakage_canary"].encode()
        leakage_files = [p.name for p in root.iterdir() if p.is_file() and canary in p.read_bytes()]
        events = [json.loads(line) for line in (root / "trajectory.jsonl").read_text().splitlines()]
        elapsed = (
            datetime.fromisoformat(events[-1]["time"]) - datetime.fromisoformat(events[0]["time"])
        ).total_seconds()
        row = {
            "task_id": task_id,
            "event_id": case["event_id"],
            "status": run["status"],
            "stop_reason": run["stop_reason"],
            "model_calls": run["budget"]["model_calls"],
            "search_read_calls": run["budget"]["tool_calls"],
            "tokens": run["budget"]["tokens"],
            "unknown_usage_calls": run["budget"]["unknown_usage_calls"],
            "elapsed_seconds": elapsed,
            "api_cost_usd": run["api_cost_usd"],
            "valid_unique_citations": len(refs & passages),
            "invalid_unique_citations": sorted(refs - passages),
            "removed_invalid_citations": run["invalid_citations"],
            "historical_boundary_errors": boundary_errors,
            "gold_canary_files": leakage_files,
            "saved_claims": len(read_json(root / "claims.json")),
            "timeline_entries": len(read_json(root / "timeline.json")),
            "read_versions": sorted(evidence["documents"]),
            **semantic,
        }
        rows.append(row)
    truth = {
        label: sum(r["truth_counts"][label] for r in rows)
        for label in ["correct", "incorrect", "uncertain"]
    }
    pairs = {
        label: sum(r["citation_pair_counts"][label] for r in rows)
        for label in ["supported", "unsupported", "contradicted", "uncertain"]
    }
    claims = sum(r["reviewed_claims"] for r in rows)
    questions = sum(r["reference_question_count"] for r in rows)
    summary = {
        "split": "development",
        "condition": pilot.get("condition", "baseline"),
        "model": pilot["model"],
        "review_method": annotations["review_method"],
        "independent_reviewers": 0,
        "pilot_sha256": file_hash(pilot_path),
        "gold_sha256": file_hash(gold_path),
        "annotations_sha256": file_hash(annotation_path),
        "cases": rows,
        "totals": {
            "model_calls": sum(r["model_calls"] for r in rows),
            "tokens": sum(r["tokens"] for r in rows),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in rows),
            "reviewed_claims": claims,
            "truth_counts": truth,
            "citation_pair_counts": pairs,
            "factual_correctness": ratio(truth["correct"], truth["correct"] + truth["incorrect"]),
            "citation_support": ratio(pairs["supported"], sum(pairs.values())),
            "citation_coverage": ratio(
                sum(r["claims_with_supporting_citation"] for r in rows), claims
            ),
            "answer_coverage": ratio(sum(r["reference_question_score"] for r in rows), questions),
            "reference_question_count": questions,
            "api_cost_usd": None,
        },
        "limitations": [
            "Two development events, two correlated cutoffs each; one run per case.",
            "Single assistant annotation, not blinded human consensus or semantic certification.",
            "Curated historical excerpts, not open-web research or final-test performance.",
            "Rates are descriptive; zero denominators remain null; dollars are not yet calculated.",
        ],
    }
    write_json(output / "summary.json", summary)
    columns = [
        "task_id",
        "status",
        "stop_reason",
        "model_calls",
        "tokens",
        "saved_claims",
        "reviewed_claims",
        "factual_correctness",
        "citation_support",
        "citation_coverage",
        "answer_coverage",
    ]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary
