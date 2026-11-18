"""One bounded research subprocess. Credentials stay in the local environment."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from .harness import implementation_hash, run_research
from .live import question_task
from .model_routing import PROFILE
from .schemas import BudgetConfig, ResearchTask
from .storage import read_json, write_json


async def work(root, run_dir):
    load_dotenv(root / ".env", override=False)
    request = read_json(run_dir / "web_request.json")
    fixture = request["mode"] == "fixture"
    budget_path = run_dir / "requested_budget.json"
    budget = (
        BudgetConfig.model_validate(read_json(budget_path))
        if budget_path.exists()
        else BudgetConfig()
    )
    # Preserve the source for this interactive run before any model dispatch.
    package = Path(__file__).parent
    files = [*package.glob("*.py"), *package.joinpath("web_assets").glob("*")]
    hashes = {}
    for source in files:
        relative = source.relative_to(package)
        target = run_dir / "code_snapshot/src/news_agent" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ["pyproject.toml", "uv.lock", "configs/base.json"]:
        source = root / name
        if source.exists():
            target = run_dir / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    write_json(
        run_dir / "web_version.json",
        {
            "implementation_hash": implementation_hash(),
            "package_files_sha256": hashes,
        },
    )
    task = (
        ResearchTask.model_validate(read_json(root / "data/fixtures/task.json"))
        if fixture
        else question_task(request["question"])
    )
    model_profile = request.get("model_profile", "auto")
    if model_profile == "auto":
        model_profile = (
            PROFILE
            if not fixture
            and request.get("workflow") == "adaptive-event-map"
            and os.environ.get("NEWS_AGENT_MODEL", "").startswith("deepseek:")
            else "single"
        )
    job = asyncio.create_task(
        run_research(
            run_dir,
            task,
            root / "data/fixtures/corpus.json" if fixture else None,
            budget,
            mock=fixture,
            model_name=None if fixture else os.environ.get("NEWS_AGENT_MODEL"),
            adapter="local" if fixture else "tavily",
            condition=request["condition"],
            visual_progress=True,
            workflow=request.get("workflow", "legacy"),
            model_profile=model_profile,
        )
    )
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, job.cancel)
    try:
        result = await job
        write_json(run_dir / "worker_status.json", {"status": result["status"]})
    except Exception as exc:
        write_json(
            run_dir / "worker_status.json", {"status": "failed", "error_type": type(exc).__name__}
        )
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


if __name__ == "__main__":
    asyncio.run(work(Path(sys.argv[1]), Path(sys.argv[2])))
