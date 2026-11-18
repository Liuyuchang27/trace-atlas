"""One balanced, serial development matrix. No scoring references are read here."""

from __future__ import annotations

from datetime import UTC, datetime

from .harness import file_hash, implementation_hash, run_research
from .mechanisms import CONDITIONS
from .retrieval import LocalCorpus
from .schemas import ResearchTask
from .storage import read_json, write_json


async def run_matrix(manifest_path, project_root, output_dir, budget, model_name, mock=False):
    if (output_dir / "matrix.json").exists() or (output_dir / "cases").exists():
        raise ValueError("matrix output already exists; preserve paid results")
    if not mock and not model_name:
        raise ValueError("a model is required for a paid matrix")
    dataset = read_json(manifest_path)
    if dataset.get("split") != "development" or not dataset.get("cases"):
        raise ValueError("matrix accepts a nonempty development manifest only")
    inputs = []
    ids = set()
    for case in dataset["cases"]:
        task_path, corpus_path = [project_root / case[k] for k in ["task_path", "corpus_path"]]
        if (
            file_hash(task_path) != case["task_sha256"]
            or file_hash(corpus_path) != case["corpus_sha256"]
        ):
            raise ValueError("frozen matrix input changed")
        task = ResearchTask.model_validate(read_json(task_path))
        if task.task_id != case["task_id"] or task.task_id in ids:
            raise ValueError("duplicate or mismatched matrix task ID")
        ids.add(task.task_id)
        if set(LocalCorpus.load(corpus_path, task).documents) != set(case["eligible_version_ids"]):
            raise ValueError("matrix eligibility differs from manifest")
        inputs.append((case, task, corpus_path))
    record = {
        "split": "development",
        "dataset_version": dataset["dataset_version"],
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": file_hash(manifest_path),
        "implementation_hash": implementation_hash(),
        "model": model_name,
        "mock": mock,
        "conditions": list(CONDITIONS),
        "repetitions": 1,
        "budget": budget.model_dump(),
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "cases": [],
        "run_order": [],
    }
    write_json(output_dir / "matrix.json", record)
    pilots = {
        condition: {**record, "condition": condition, "cases": [], "run_order": []}
        for condition in CONDITIONS
    }
    for index, (case, task, corpus_path) in enumerate(inputs):
        order = [*CONDITIONS[index % 4 :], *CONDITIONS[: index % 4]]
        for condition in order:
            if implementation_hash() != record["implementation_hash"]:
                raise ValueError("implementation changed during the matrix")
            run_dir = output_dir / "cases" / task.task_id / condition
            result = await run_research(
                run_dir,
                task,
                corpus_path,
                budget,
                mock=mock,
                model_name=model_name,
                condition=condition,
            )
            saved = {
                "task_id": task.task_id,
                "event_id": case["event_id"],
                "condition": condition,
                "run_dir": str(run_dir.resolve()),
                "status": result["status"],
                "stop_reason": result["stop_reason"],
            }
            record["cases"].append(saved)
            record["run_order"].append({"task_id": task.task_id, "condition": condition})
            pilots[condition]["cases"].append(saved)
            pilots[condition]["run_order"].append({"task_id": task.task_id, "condition": condition})
            write_json(output_dir / "matrix.json", record)
            write_json(output_dir / "conditions" / condition / "pilot.json", pilots[condition])
    record.update(status="finished", finished_at=datetime.now(UTC).isoformat())
    write_json(output_dir / "matrix.json", record)
    for condition, pilot in pilots.items():
        pilot.update(status="finished", finished_at=record["finished_at"])
        write_json(output_dir / "conditions" / condition / "pilot.json", pilot)
    return record
