"""Serial, frozen-input baseline development runs. No scoring references are loaded here."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .harness import file_hash, implementation_hash, run_research
from .mechanisms import CONDITIONS
from .retrieval import LocalCorpus
from .schemas import BudgetConfig, ResearchTask
from .storage import read_json, write_json


async def run_pilot(
    manifest_path: Path,
    project_root: Path,
    output_dir: Path,
    budget: BudgetConfig,
    model_name: str | None,
    mock: bool = False,
    condition: str = "baseline",
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError("unknown experiment condition")
    dataset = read_json(manifest_path)
    if dataset.get("split") != "development":
        raise ValueError("pilot accepts development data only")
    if (output_dir / "pilot.json").exists():
        raise ValueError("pilot output already exists; inspect it instead of rerunning paid cases")
    if not mock and not model_name:
        raise ValueError("provide a model before starting a paid pilot")
    # Validate every input before any paid model call.
    inputs = []
    task_ids = set()
    for case in dataset["cases"]:
        task_path, corpus_path = (project_root / case[k] for k in ["task_path", "corpus_path"])
        if (
            file_hash(task_path) != case["task_sha256"]
            or file_hash(corpus_path) != case["corpus_sha256"]
        ):
            raise ValueError(f"frozen development input changed: {case['task_id']}")
        task = ResearchTask.model_validate(read_json(task_path))
        if task.task_id != case["task_id"]:
            raise ValueError("task ID disagrees with manifest")
        if task.task_id in task_ids:
            raise ValueError("duplicate development task ID")
        task_ids.add(task.task_id)
        visible = set(LocalCorpus.load(corpus_path, task).documents)
        if visible != set(case["eligible_version_ids"]):
            raise ValueError("historical eligibility disagrees with frozen manifest")
        inputs.append((case, task, corpus_path))
    record = {
        "dataset_version": dataset["dataset_version"],
        "split": "development",
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": file_hash(manifest_path),
        "implementation_hash": implementation_hash(),
        "model": model_name,
        "mock": mock,
        "condition": condition,
        "budget": budget.model_dump(),
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "cases": [],
    }
    write_json(output_dir / "pilot.json", record)
    for case, task, corpus_path in inputs:
        if implementation_hash() != record["implementation_hash"]:
            raise ValueError("implementation changed during the pilot")
        run_dir = output_dir / "cases" / case["task_id"]
        result = await run_research(
            run_dir,
            task,
            corpus_path,
            budget,
            mock=mock,
            model_name=model_name,
            condition=condition,
        )
        record["cases"].append(
            {
                "task_id": case["task_id"],
                "event_id": case["event_id"],
                "run_dir": str(run_dir.resolve()),
                "status": result["status"],
                "stop_reason": result["stop_reason"],
            }
        )
        write_json(output_dir / "pilot.json", record)
    record.update(status="finished", finished_at=datetime.now(UTC).isoformat())
    write_json(output_dir / "pilot.json", record)
    return record
