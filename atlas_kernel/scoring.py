"""Score completed predictions with separately installed benchmark toolkits."""
import argparse
import ast
from collections import defaultdict
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tls(repo, case, rows):
    sys.path.insert(0, str(repo))
    from evaluation import evaluate_dates
    from tilse.data.timelines import GroundTruth, Timeline
    from tilse.evaluation.rouge import TimelineRougeEvaluator

    def day(value):
        parts = str(value).split("T")[0].strip().split("-")
        return date.fromisoformat("-".join((parts + ["01","01"])[:3]))

    gold = []
    reference = repo / "data" / "open" / case / "timelines.jsonl"
    for line in reference.read_text().splitlines():
        if line.strip() and (items := ast.literal_eval(line)):
            gold.append(Timeline({day(stamp):entries for stamp,entries in items}))
    if not gold:
        raise ValueError("Missing reference timelines: " + case)
    if not rows:
        zero = dict(precision=0., recall=0., f_score=0.)
        return {"dates":zero, "align_date_content_costs":{"rouge_1":zero,"rouge_2":zero}}
    pred = Timeline({day(r["start"]):r["events"] for r in rows})
    truth = GroundTruth(gold)
    metric = TimelineRougeEvaluator(measures=["rouge_1","rouge_2"])
    return {"dates":evaluate_dates(pred,truth),"align_date_content_costs":metric.evaluate_align_date_content_costs(pred,truth)}


def report_commands(args, submission):
    repo = args.repo.resolve()
    query = repo / "data/prompt_data/query.jsonl"
    if not query.is_file():
        raise ValueError("Expected an official DeepResearch Bench checkout containing the query file")
    output = args.out.resolve()
    if args.judge in {"both","race"}:
        yield ["deepresearch_bench_race.py",submission.stem,"--query_file",query,"--raw_data_dir",submission.parent,
               "--output_dir",output/"race","--cleaned_data_dir",output/"cleaned"]
    if args.judge in {"both","fact"}:
        previous = submission
        for phase in ["extract","deduplicate","scrape","validate"]:
            destination = output / ("fact-"+phase+".jsonl")
            flags = [] if phase == "scrape" else ["--query_data_path",query]
            yield ["-m","utils."+phase,"--raw_data_path",previous,"--output_path",destination,*flags]
            previous = destination
        yield ["-m","utils.stat","--input_path",previous,"--output_path",output/"fact.txt"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite",choices=["open-tls","deepresearch-bench","browsecomp-plus"])
    parser.add_argument("runs",type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--repo",type=Path,help="Official CHRONOS or DeepResearch Bench checkout")
    parser.add_argument("--qrels",type=Path,help="Separate held-out relevance judgements")
    parser.add_argument("--judge",choices=["race","fact","both"])
    parser.add_argument("--plan",action="store_true",help="Write official judge commands and submission without calling paid services")
    parser.add_argument("--judge-python",default=sys.executable)
    parser.add_argument("--judge-env-file",type=Path,help="Local judge credentials for live RACE/FACT scoring")
    args=parser.parse_args()
    if args.suite=="open-tls" and not args.repo:
        parser.error("--repo must point to CHRONOS")
    if args.judge and (args.suite!="deepresearch-bench" or not args.repo):
        parser.error("--judge requires deepresearch-bench and --repo")
    if args.plan and not args.judge:
        parser.error("--plan requires --judge")
    files = [args.runs/"prediction.json"] if (args.runs/"prediction.json").is_file() else sorted(args.runs.glob("*/prediction.json"))
    skipped = [str(p.parent.name) for p in args.runs.glob("*/run.json") if not (p.parent/"prediction.json").exists()]
    if skipped:
        parser.error("Missing predictions for these runs: "+", ".join(skipped))
    if not files:
        parser.error("No predictions in the supplied directory")
    rows = {p.parent.name:json.loads(p.read_text()) for p in files}
    args.out.mkdir(exist_ok=True,parents=True)
    if args.suite=="open-tls":
        result = {case:tls(args.repo.resolve(),case,row["predict-timeline"]) for case,row in rows.items()}
        write(args.out/"scores.json",result)
    elif args.suite=="browsecomp-plus":
        for case,row in rows.items():
            write(args.out/(case+".json"),row)
        if args.qrels:
            judged = defaultdict(set)
            for line in args.qrels.read_text().splitlines():
                if line.strip():
                    case,_,doc,score = line.split()
                    if int(score)>0:
                        judged[case].add(doc)
            result={}
            for case,row in rows.items():
                gold=judged[case]
                if not gold:
                    raise ValueError("Missing qrels for " + case)
                result[case] = {key:len(set(row[key]) & gold)/len(gold) for key in ("retrieved_docids","opened_docids")}
            write(args.out.parent/(args.out.name+"-recall.json"),result)
    else:
        submission=(args.out/"trace-atlas.jsonl").resolve()
        submission.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows.values()),encoding="utf-8")
        if args.judge:
            commands=[[os.path.abspath(args.judge_python),"-B",*map(str,cmd)] for cmd in report_commands(args,submission)]
            write(args.out/"judge-commands.json",commands)
            if not args.plan:
                if args.judge_env_file:
                    if not args.judge_env_file.is_file():
                        parser.error("Judge environment file does not exist")
                    from dotenv import load_dotenv
                    load_dotenv(args.judge_env_file)
                backend = os.environ.setdefault("LLM_BACKEND", "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai")
                if backend not in {"openai","openrouter"} or not os.getenv(backend.upper()+"_API_KEY"):
                    parser.error("Configure an OpenAI or OpenRouter judge credential")
                if args.judge in {"fact","both"} and not os.getenv("JINA_API_KEY"):
                    parser.error("FACT requires JINA_API_KEY")
                for cmd in commands:
                    subprocess.run(cmd,cwd=args.repo,check=True)


if __name__=="__main__":
    main()
