from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from .evaluation import evaluate_pilot
from .harness import run_research
from .live import question_task
from .matrix import run_matrix
from .mechanisms import CONDITIONS
from .model_routing import PROFILE
from .pilot import run_pilot
from .schemas import BudgetConfig, ResearchTask
from .storage import read_json


def main(argv=None) -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="News research H0 on the actual Deep Agents SDK")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Start a new research run")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--corpus", type=Path)
    run.add_argument("--adapter", choices=["local", "tavily"], default="local")
    run.add_argument("--condition", choices=CONDITIONS, default="baseline")
    run.add_argument(
        "--workflow", choices=["legacy", "event-map", "adaptive-event-map"], default="legacy"
    )
    run.add_argument("--mock", action="store_true")
    run.add_argument("--model")
    run.add_argument("--model-profile", choices=["single", PROFILE])
    run.add_argument("--budget", type=Path, default=None)
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ask = commands.add_parser("ask", help="Let the agent search/read the web for your question")
    ask.add_argument("question")
    ask.add_argument("--event", help="Optional topic label; no source URLs or prepared corpus")
    ask.add_argument("--timezone", default="Asia/Hong_Kong")
    ask.add_argument("--condition", choices=CONDITIONS, default="baseline")
    ask.add_argument(
        "--workflow", choices=["legacy", "event-map", "adaptive-event-map"], default="legacy"
    )
    ask.add_argument("--model")
    ask.add_argument("--model-profile", choices=["single", PROFILE])
    ask.add_argument("--budget", type=Path)
    ask.add_argument("--runs-dir", type=Path, default=Path("runs/live-web"))
    ask.set_defaults(adapter="tavily", mock=False, corpus=None)
    ask.add_argument("--visual-progress", action="store_true", help="Record public plan/gap steps")
    web = commands.add_parser("serve", help="Open a local visual test app for the research agent")
    web.add_argument("--port", type=int, default=8768)
    web.add_argument("--budget", type=Path, help="Budget configuration for new live investigations")
    web.add_argument("--project-root", type=Path, default=Path.cwd())
    resume = commands.add_parser("resume", help="Resume the same checkpoint and original budget")
    resume.add_argument("--run-dir", type=Path, required=True)
    pilot = commands.add_parser("pilot", help="Run frozen development baseline cases serially")
    pilot.add_argument("--manifest", type=Path, default=Path("data/dev/manifest.json"))
    pilot.add_argument("--project-root", type=Path, default=Path.cwd())
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument("--budget", type=Path, default=Path("configs/base.json"))
    pilot.add_argument("--model")
    pilot.add_argument("--mock", action="store_true")
    pilot.add_argument("--condition", choices=CONDITIONS, default="baseline")
    matrix = commands.add_parser("matrix", help="Run a balanced four-condition development pilot")
    matrix.add_argument("--manifest", type=Path, default=Path("data/dev/manifest.json"))
    matrix.add_argument("--project-root", type=Path, default=Path.cwd())
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--budget", type=Path, default=Path("configs/base.json"))
    matrix.add_argument("--model")
    matrix.add_argument("--mock", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Score a dev pilot using reviewed annotations")
    evaluate.add_argument("--pilot", type=Path, required=True)
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--annotations", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        from .web import serve

        budget = BudgetConfig.model_validate(read_json(args.budget)) if args.budget else None
        serve(args.project_root.resolve(), args.port, budget=budget)
        return 0
    try:
        if args.command == "evaluate":
            result = evaluate_pilot(args.pilot, args.gold, args.annotations, args.output_dir)
        elif args.command in {"pilot", "matrix"}:
            runner = run_matrix if args.command == "matrix" else run_pilot
            options = {} if args.command == "matrix" else {"condition": args.condition}
            result = asyncio.run(
                runner(
                    args.manifest.resolve(),
                    args.project_root.resolve(),
                    args.output_dir.resolve(),
                    BudgetConfig.model_validate(read_json(args.budget)),
                    args.model or (None if args.mock else os.environ.get("NEWS_AGENT_MODEL")),
                    mock=args.mock,
                    **options,
                )
            )
        elif args.command == "resume":
            result = asyncio.run(run_research(args.run_dir.resolve(), resume=True))
        else:
            if args.mock and args.adapter != "local":
                raise ValueError("--mock requires a local corpus and performs no external requests")
            if args.mock and args.model:
                raise ValueError("use either --mock or --model, not both")
            task = (
                question_task(args.question, args.event, args.timezone)
                if args.command == "ask"
                else ResearchTask.model_validate(read_json(args.task))
            )
            budget = (
                BudgetConfig.model_validate(read_json(args.budget))
                if args.budget
                else BudgetConfig()
            )
            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
            result = asyncio.run(
                run_research(
                    run_dir=args.runs_dir.resolve() / run_id,
                    task=task,
                    corpus_path=args.corpus,
                    budget_config=budget,
                    mock=args.mock,
                    model_name=args.model
                    or (None if args.mock else os.environ.get("NEWS_AGENT_MODEL")),
                    adapter=args.adapter,
                    condition=args.condition,
                    visual_progress=getattr(args, "visual_progress", False),
                    workflow=args.workflow,
                    model_profile=args.model_profile,
                )
            )
        displayed = result
        if args.command == "ask":
            displayed = {
                k: result[k]
                for k in [
                    "run_id",
                    "condition",
                    "status",
                    "stop_reason",
                    "model_finalized",
                    "report",
                    "activity",
                ]
            }
            displayed["usage"] = {
                k: result["budget"].get(k, 0)
                for k in [
                    "model_calls",
                    "tokens",
                    "web_search_requests",
                    "web_extract_requests",
                    "retrieval_credits",
                    "retrieval_unknown_usage_calls",
                ]
            }
        print(json.dumps(displayed, ensure_ascii=False, indent=2))
        return int(
            result.get("status") == "failed"
            or any(c["status"] == "failed" for c in result.get("cases", []) if "status" in c)
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
