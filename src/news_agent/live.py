"""Live user-question entry point and a journal-derived activity report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .schemas import ResearchTask


def question_task(question, event=None, timezone="Asia/Hong_Kong"):
    if not question.strip():
        raise ValueError("provide a nonempty research question")
    try:
        local_zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("unknown timezone") from exc
    now = datetime.now(UTC)
    observation = {
        "utc": now.isoformat(),
        "local": now.astimezone(local_zone).isoformat(),
        "timezone": timezone,
        "scope": "live retrieval observed during the run; not verified historical snapshots",
    }
    return ResearchTask(
        task_id="live-" + now.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8],
        event=event or question.strip(),
        question=question.strip()
        + "\n\nRun observation context: "
        + json.dumps(observation, ensure_ascii=False),
        mode="retrospective",
        synthetic=False,
    )


def export_activity(store, outcome):
    events = [
        json.loads(line) for line in (store.root / "trajectory.jsonl").read_text().splitlines()
    ]
    lines = [
        "# Agent live-web activity",
        "",
        "Queries and URLs below come from the project model's tool calls.",
        "No prepared corpus, seed URLs or answer were supplied to this run.",
        "",
        f"- Status: {outcome['status']} / {outcome['stop_reason']}",
        f"- Model calls: {outcome['budget']['model_calls']}",
        f"- Model accounted tokens: {outcome['budget']['tokens']}",
        f"- Web requests: {outcome['budget'].get('retrieval_api_calls', 0)}",
        f"- Report: {outcome['report']}",
        "",
        "## Actual tool and network activity",
        "",
    ]
    timeline = []
    for e in events:
        row = {"sequence": e["sequence"], "time": e["time"], "kind": e["kind"]}
        if e["kind"] == "tool_dispatch" and e["name"] in {"search", "read_document", "open_url"}:
            row.update(name=e["name"], arguments=e.get("arguments", {}))
        elif e["kind"] == "retrieval_dispatch":
            row.update(
                operation=e["operation"], request_id=e["request_id"], arguments=e["arguments"]
            )
        elif e["kind"] == "retrieval_result":
            row.update(
                {
                    k: e.get(k)
                    for k in [
                        "operation",
                        "request_id",
                        "status_code",
                        "error",
                        "credits",
                        "usage_known",
                    ]
                }
            )
        elif e["kind"] == "web_page_saved":
            row.update(
                {
                    k: e[k]
                    for k in [
                        "version_id",
                        "url",
                        "content_sha256",
                        "saved_characters",
                        "truncated",
                    ]
                }
            )
        else:
            continue
        timeline.append(row)
        description = json.dumps(
            {k: v for k, v in row.items() if k not in {"sequence", "time"}}, ensure_ascii=False
        )
        lines.append(f"- {row['sequence']}. {row['time']} `{description}`")
    lines += [
        "",
        "Search snippets are not evidence passages. Only pages selected and extracted are citable.",
        "Current web extraction does not establish historical availability. "
        "Model assessments do not certify facts.",
    ]
    store.save(
        "live_activity.json",
        {"method": "append-only execution journal, no model self-report", "events": timeline},
    )
    (store.root / "activity.md").write_text("\n".join(lines) + "\n")
    return str(store.root / "activity.md")
