import asyncio
from pathlib import Path

import pytest

import news_agent.matrix as matrix
from news_agent.harness import file_hash
from news_agent.retrieval import LocalCorpus
from news_agent.schemas import BudgetConfig
from news_agent.storage import read_json, write_json


def manifest(tmp_path, task, corpus_path):
    cases = []
    for index in range(4):
        copied = task.model_copy(update={"task_id": f"fixture-{index}"})
        path = tmp_path / f"task-{index}.json"
        write_json(path, copied.model_dump(mode="json"))
        cases.append(
            {
                "task_id": copied.task_id,
                "event_id": "synthetic-acquisition",
                "task_path": str(path),
                "corpus_path": str(corpus_path),
                "task_sha256": file_hash(path),
                "corpus_sha256": file_hash(corpus_path),
                "eligible_version_ids": list(LocalCorpus.load(corpus_path, task).documents),
            }
        )
    path = tmp_path / "manifest.json"
    write_json(path, {"split": "development", "dataset_version": "synthetic-only", "cases": cases})
    return path


def test_balanced_matrix_runs_actual_sdk_and_preserves_all_conditions(tmp_path, task, corpus_path):
    path = manifest(tmp_path, task, corpus_path)
    output = tmp_path / "matrix"
    result = asyncio.run(matrix.run_matrix(path, tmp_path, output, BudgetConfig(), None, mock=True))
    assert result["status"] == "finished"
    assert len(result["cases"]) == 16
    assert all(case["status"] != "failed" for case in result["cases"])
    for position in range(4):
        assert {result["run_order"][index * 4 + position]["condition"] for index in range(4)} == {
            "baseline",
            "t",
            "s",
            "ts",
        }
    for condition in ["baseline", "t", "s", "ts"]:
        pilot = read_json(output / "conditions" / condition / "pilot.json")
        assert len(pilot["cases"]) == len(pilot["run_order"]) == 4
        assert pilot["condition"] == condition
        assert all(c["condition"] == condition for c in pilot["cases"])
        for case in pilot["cases"]:
            run = read_json(Path(case["run_dir"]) / "run.json")
            assert run["model_finalized"]
            assert run["condition"] == condition
            if condition == "baseline":
                assert not (Path(case["run_dir"]) / "investigation.json").exists()
            else:
                assert run["mechanisms"]["review_model_calls"] > 0
    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(matrix.run_matrix(path, tmp_path, output, BudgetConfig(), None, mock=True))


def test_matrix_validates_every_input_before_any_paid_call(
    tmp_path, task, corpus_path, monkeypatch
):
    path = manifest(tmp_path, task, corpus_path)
    value = read_json(path)
    value["cases"][-1]["corpus_sha256"] = "bad-digest"
    write_json(path, value)

    async def forbidden(*args, **kwargs):
        raise AssertionError("paid model called before all inputs were validated")

    monkeypatch.setattr(matrix, "run_research", forbidden)
    with pytest.raises(ValueError, match="input changed"):
        asyncio.run(
            matrix.run_matrix(
                path, tmp_path, tmp_path / "output", BudgetConfig(), "deepseek:deepseek-flash"
            )
        )
    assert not (tmp_path / "output" / "matrix.json").exists()
