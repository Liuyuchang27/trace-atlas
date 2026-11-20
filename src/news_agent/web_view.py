"""Read-only projections of real journals into a graph, timeline and linked report."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from .event_view import event_projection
from .report_presentation import current_brief, render_brief, report_brief, verified_review
from .storage import read_json
from .time_display import DATE_NOTE

PIDS = re.compile(r"p-[a-f0-9]{16}")


def maybe_json(path, default=None):
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def journal(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a concurrent append may not have flushed the last line yet
    return rows


def safe_url(url):
    try:
        parsed = urlsplit(url or "")
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username:
            return url
    except ValueError:
        pass
    return None


def run_key(root, run_dir):
    return hashlib.sha256(str(run_dir.relative_to(root)).encode()).hexdigest()[:24]


KINDS = {
    "begin_temporal_context": "read",
    "propose_temporal_resolution": "verify",
    "retain_temporal_unknown": "gap",
    "request_temporal_reaudit": "verify",
    "begin_investigation": "plan",
    "replan_gap": "plan",
    "search_gap": "search",
    "collect_monthly_history": "search",
    "select_monthly_series": "read",
    "search_current_sources": "search",
    "read_current_sources": "read",
    "refresh_current_status": "search",
    "search_previous_sales": "search",
    "record_current_result": "verify",
    "append_event_finding": "claim",
    "append_undated_finding": "claim",
    "create_event_node": "claim",
    "close_gap_round": "gap",
    "review_node_sources": "source_check",
    "resolve_event_write": "verify",
    "set_investigation_scope": "plan",
    "record_event_finding": "claim",
    "record_event_relation": "verify",
    "record_source_relation": "source_check",
    "inspect_event_graph": "read",
    "search": "search",
    "read_document": "read",
    "open_url": "read",
    "get_passage": "read",
    "find_saved_text": "read",
    "update_question_evidence": "verify",
    "list_source_links": "read",
    "record_claim": "claim",
    "verify_claim": "verify",
    "trace_origin": "source_check",
    "seek_corroboration": "followup",
    "focus_date_or_subevent": "temporal",
    "extend_time_window": "temporal",
    "check_event_transition": "verify",
    "write_todos": "plan",
    "publish_plan": "plan",
    "list_evidence": "claim",
}
LABELS = {
    "begin_temporal_context": "回读时间依据上下文",
    "propose_temporal_resolution": "关联时间补证与旧缺口",
    "retain_temporal_unknown": "补证后保留时间未知",
    "request_temporal_reaudit": "提交独立时间复审",
    "begin_investigation": "确定范围与核心问题",
    "replan_gap": "根据反馈调整方案",
    "search_gap": "按缺口补搜",
    "collect_monthly_history": "查找完整月度原表",
    "select_monthly_series": "选定并读取月度序列",
    "search_current_sources": "查最新一手消息和目标数据",
    "read_current_sources": "核对一手原文",
    "refresh_current_status": "发现预载旧记录，直查官方公告",
    "search_previous_sales": "回查上一披露时段",
    "record_current_result": "给出截至核查日的结论",
    "append_event_finding": "证据写回节点",
    "append_undated_finding": "保留未知日期的证据",
    "create_event_node": "建立事件分支",
    "close_gap_round": "核对本轮增益",
    "review_node_sources": "审视原始出处",
    "resolve_event_write": "纠正证据绑定",
    "set_investigation_scope": "设定时间追查方向",
    "record_event_finding": "记录事件与原文",
    "record_event_relation": "关联转折与解释",
    "record_source_relation": "比对来源关系",
    "inspect_event_graph": "查看事件状态历史",
    "search": "搜索",
    "read_document": "读取网页",
    "open_url": "打开链接",
    "get_passage": "重读证据",
    "find_saved_text": "定位已选页正文",
    "update_question_evidence": "关联证据与更新问题",
    "list_source_links": "查找原文来源链接",
    "record_claim": "保存主张",
    "verify_claim": "核验主张",
    "trace_origin": "追溯来源",
    "seek_corroboration": "针对缺口补证",
    "focus_date_or_subevent": "聚焦时间 / 子事件",
    "extend_time_window": "扩展时间窗口",
    "check_event_transition": "核查事件阶段",
    "write_todos": "更新任务计划",
    "publish_plan": "发布调查计划",
    "list_evidence": "整理已有证据",
}


def build_snapshot(run_dir: Path, *, active=False, worker_status=None):
    config = maybe_json(run_dir / "effective_config.json")
    task = maybe_json(run_dir / "task.json")
    request = maybe_json(run_dir / "web_request.json")
    result = maybe_json(run_dir / "run.json")
    budget = maybe_json(run_dir / "budget.json") or result.get("budget", {})
    events = journal(run_dir / "trajectory.jsonl")
    evidence = maybe_json(run_dir / "evidence.json")
    docs = evidence.get("documents", {})
    passages = evidence.get("passages", {})
    cache = maybe_json(run_dir / "tool_cache.json")
    candidates = maybe_json(run_dir / "retrieval_cache.json").get("candidates", {})
    investigation = maybe_json(run_dir / "investigation.json")
    question = task.get("question", request.get("question", "正在准备任务"))
    question = question.split("\n\nRun observation context:")[0]
    root_node = {
        "id": "task",
        "kind": "task",
        "title": question,
        "status": "running" if active else result.get("status", worker_status or "pending"),
        "sequence": 0,
        "provenance": "user_request",
        "details": {"mode": task.get("mode")},
    }
    nodes, edges, action_nodes = {"task": root_node}, [], {}
    previous = "task"
    network_requests = []
    request_nodes = {}
    current_action = None
    repair_step = None
    temporal_audit = None
    retry_nodes = {}
    repair_parents = {}
    for event in events:
        kind, seq = event["kind"], event["sequence"]
        if kind == "public_plan_updated":
            for step in event["steps"]:
                identifier = "plan:" + step["step_id"]
                update = {**step, "sequence": seq, "time": event["time"]}
                if identifier not in nodes:
                    nodes[identifier] = {
                        "id": identifier,
                        "sequence": seq,
                        "kind": step["kind"],
                        "title": step["title"],
                        "status": step["status"],
                        "details": step,
                        "provenance": "model_reported",
                        "updates": [],
                    }
                nodes[identifier]["updates"].append(update)
                parent = "plan:" + step["parent_id"] if step.get("parent_id") else "task"
                edge = {"from": parent, "to": identifier, "kind": "explicit", "sequence": seq}
                if edge not in edges:
                    edges.append(edge)
        elif kind == "tool_dispatch":
            current_action = None
            name, args = event["name"], event.get("arguments", {})
            identifier = "tool:" + event["action_id"]
            current_action = identifier
            payload = {}
            try:
                payload = json.loads(cache.get(event["action_id"], {}).get("content", "{}"))
            except (ValueError, TypeError):
                pass
            if not isinstance(payload, dict):
                payload = {"value": payload}
            title = args.get("query") or args.get("text") or args.get("url")
            doc = docs.get(args.get("version_id"), candidates.get(args.get("version_id"), {}))
            title = title or doc.get("title") or LABELS.get(name, name)
            nodes[identifier] = {
                "id": identifier,
                "kind": KINDS.get(name, "verify"),
                "title": title,
                "label": LABELS.get(name, name),
                "status": "running",
                "sequence": seq,
                "time": event["time"],
                "provenance": "actual_tool_call",
                "tool": name,
                "details": {"arguments": args, "result": payload},
                "url": safe_url(args.get("url") or doc.get("url")),
                "passage_ids": list(dict.fromkeys(PIDS.findall(json.dumps(payload)))),
            }
            action_nodes[event["action_id"]] = identifier
            if event["action_id"] in repair_parents:
                edges.append(
                    {
                        "from": repair_parents[event["action_id"]],
                        "to": identifier,
                        "kind": "explicit",
                        "sequence": seq,
                    }
                )
            edges.append({"from": previous, "to": identifier, "kind": "sequence", "sequence": seq})
            if event.get("public_step_id"):
                edges.append(
                    {
                        "from": "plan:" + event["public_step_id"],
                        "to": identifier,
                        "kind": "context",
                        "sequence": seq,
                    }
                )
            previous = identifier
        elif kind == "adaptive_search_dispatch" and event.get("action_id") in action_nodes:
            node = nodes[action_nodes[event["action_id"]]]
            if event.get("target_question"):
                node["details"]["target"] = {
                    k: event[k]
                    for k in (
                        "target_question",
                        "question_part_id",
                        "parent_gap_id",
                        "reference_month",
                        "window",
                        "query",
                    )
                    if k in event
                }
                if month := event.get("reference_month"):
                    node["label"] = "按月补查"
                    node["title"] = f"{month} · {node['title']}"
                else:
                    node["label"] = "按分项补查"
        elif kind == "search_intent" and current_action:
            node = nodes[current_action]
            node["public_reason"] = event["purpose"]
            node["details"]["public_intent"] = event
            identifier = "intent:" + str(seq)
            intent_kind = {
                "initial": "plan",
                "gap": "gap",
                "source_trace": "source_check",
                "temporal": "temporal",
                "followup": "followup",
            }[event["intent"]]
            nodes[identifier] = {
                "id": identifier,
                "kind": intent_kind,
                "title": event["purpose"],
                "reason": event["purpose"],
                "status": "planned",
                "sequence": seq,
                "provenance": "model_reported",
                "details": event,
                "passage_ids": event.get("based_on", []),
            }
            parent = "plan:" + event["step_id"] if event.get("step_id") else "task"
            edges.extend(
                [
                    {"from": parent, "to": identifier, "kind": "explicit", "sequence": seq},
                    {"from": identifier, "to": current_action, "kind": "explicit", "sequence": seq},
                ]
            )
        elif kind == "tool_result" and event["action_id"] in action_nodes:
            node = nodes[action_nodes[event["action_id"]]]
            payload = node["details"]["result"]
            node.update(
                end_sequence=seq,
                final_status="blocked"
                if payload.get("error") or payload.get("status") == "proposal_rejected"
                else "completed",
            )
        elif kind in {
            "temporal_review_started",
            "temporal_target_deferred",
            "adaptive_gap_deferred",
            "adaptive_replan_triggered",
        }:
            identifier = "event:" + str(seq)
            is_audit = kind == "temporal_review_started"
            if is_audit:
                temporal_audit = identifier
            nodes[identifier] = {
                "id": identifier,
                "kind": "verify" if is_audit else "gap",
                "title": {
                    "temporal_review_started": "独立复审日期与问题覆盖",
                    "adaptive_replan_triggered": "本轮证据不足，重新规划",
                }.get(kind, "搁置当前缺口，继续调查"),
                "status": "running" if is_audit else "blocked",
                "sequence": seq,
                "time": event["time"],
                "details": event,
                "provenance": "runtime_event",
            }
            edges.append({"from": previous, "to": identifier, "kind": "sequence", "sequence": seq})
            previous = identifier
        elif kind == "temporal_review_finished" and temporal_audit:
            nodes[temporal_audit].update(
                end_sequence=seq,
                final_status="completed" if event["status"] == "reviewed" else "blocked",
            )
            nodes[temporal_audit]["details"] = {
                "arguments": nodes[temporal_audit]["details"],
                "result": event,
            }
            temporal_audit = None
        elif kind == "model_retry_scheduled":
            identifier = "retry:" + event["retry_id"]
            node = {
                "id": identifier,
                "kind": "verify",
                "label": "连接重试",
                "title": f"DeepSeek 重试 {event['retry_number']}/{event['max_retries']}",
                "status": "waiting",
                "sequence": seq,
                "time": event["time"],
                "details": {"arguments": event},
                "provenance": "runtime_event",
            }
            nodes[identifier] = retry_nodes[event["retry_id"]] = node
            edges.append({"from": previous, "to": identifier, "kind": "sequence", "sequence": seq})
            previous = identifier
        elif kind == "model_retry_started" and event["retry_id"] in retry_nodes:
            retry_nodes[event["retry_id"]]["started_sequence"] = seq
        elif kind == "model_retry_finished" and event["retry_id"] in retry_nodes:
            node = retry_nodes[event["retry_id"]]
            node["details"]["result"] = event
            node.update(
                end_sequence=seq,
                final_status="completed" if event["status"] == "succeeded" else "blocked",
            )
        elif kind in {
            "monthly_series_route_changed",
            "monthly_series_lookup_refined",
            "monthly_series_source_selected",
            "monthly_series_history_link",
            "monthly_series_source_rejected",
            "monthly_series_calculated",
            "monthly_series_unresolved",
        }:
            identifier = "series:" + str(seq)
            nodes[identifier] = {
                "id": identifier,
                "kind": "gap" if kind.endswith(("rejected", "unresolved")) else "verify",
                "title": {
                    "monthly_series_route_changed": "当前来源缺少完整表，改查统计数据库",
                    "monthly_series_lookup_refined": "目录未得到可用原表，补搜序列入口",
                    "monthly_series_source_selected": "Agent 选定统计序列",
                    "monthly_series_history_link": "在指标页找到完整历史表链接",
                    "monthly_series_source_rejected": "原表未通过口径或覆盖检查",
                    "monthly_series_calculated": "读取同版原表并计算各月环比",
                    "monthly_series_unresolved": "完整历史表仍有缺口，转为逐月检索",
                }[kind],
                "status": "blocked" if kind.endswith(("rejected", "unresolved")) else "completed",
                "sequence": seq,
                "time": event["time"],
                "details": event,
                "url": safe_url(event.get("url")),
                "provenance": "runtime_event",
            }
            if event.get("points"):
                nodes[identifier]["passage_ids"] = list(
                    dict.fromkeys(pid for point in event["points"] for pid in point["passage_ids"])
                )
            edges.append({"from": previous, "to": identifier, "kind": "sequence", "sequence": seq})
            previous = identifier
        elif kind == "retrieval_dispatch":
            # The tool boundary serializes calls, including nested gap searches.
            row = {**event, "node_id": current_action}
            network_requests.append(row)
            request_nodes[event["request_id"]] = row
            if current_action:
                nodes[current_action].setdefault("requests", []).append(row)
        elif kind == "retrieval_result":
            row = request_nodes.get(event["request_id"])
            if row is not None:
                row["result"] = event
        elif kind in {"gap_repair_step_requested", "gap_repair_not_admitted", "gap_repair_stopped"}:
            identifier = "repair-step:" + str(seq)
            repair_step = identifier
            title = (
                "有界补证：" + event["stage"]
                if kind == "gap_repair_step_requested"
                else "补证停止：" + event.get("reason", "unknown")
            )
            nodes[identifier] = {
                "id": identifier,
                "kind": "followup",
                "title": title,
                "status": "completed"
                if kind == "gap_repair_step_requested" or event.get("covered_provisionally")
                else "blocked",
                "sequence": seq,
                "time": event["time"],
                "details": event,
                "provenance": "runtime_bounded_repair_policy",
            }
            edges.append({"from": "task", "to": identifier, "kind": "explicit", "sequence": seq})
        elif kind == "gap_repair_tool_started" and repair_step:
            repair_parents[event["action_id"]] = repair_step
        elif kind == "question_state_changed":
            identifier = "question-transition:" + str(seq)
            nodes[identifier] = {
                "id": identifier,
                "kind": "verify",
                "title": (
                    f"问题 {event['question_id']}：{event['before_status']} → "
                    f"{event['after_status']}（暂定证据覆盖）"
                ),
                "status": "completed",
                "sequence": seq,
                "time": event["time"],
                "details": event,
                "provenance": "runtime_question_transition",
                "verification": "model_assessed_unverified",
            }
            parent = "tool:" + event["action_id"] if event.get("action_id") else "task"
            if parent in nodes:
                edges.append(
                    {"from": parent, "to": identifier, "kind": "explicit", "sequence": seq}
                )
        elif kind in {"research_stopped", "run_error", "report_revision_started", "run_finished"}:
            identifier = "event:" + str(seq)
            labels = {
                "research_stopped": "研究停止，保留写作额度",
                "run_error": "运行错误",
                "report_revision_started": "核对并修订报告",
                "run_finished": "交付报告",
            }
            nodes[identifier] = {
                "id": identifier,
                "kind": "synthesis",
                "title": labels[kind],
                "status": "blocked" if kind in {"run_error", "research_stopped"} else "completed",
                "sequence": seq,
                "time": event["time"],
                "details": event,
                "provenance": "runtime_event",
            }
            edges.append({"from": previous, "to": identifier, "kind": "sequence", "sequence": seq})
            previous = identifier
    # Missing results after a finished worker are interrupted, not forever running.
    for node in nodes.values():
        if node.get("status") == "running" and not active and node.get("tool"):
            if "end_sequence" not in node:
                node["final_status"] = "interrupted"
                node["end_sequence"] = events[-1]["sequence"] if events else node["sequence"]
    report_path = run_dir / "report.md"
    report = report_path.read_text() if report_path.exists() else ""
    claims = maybe_json(run_dir / "claims.json", [])
    temporal = investigation.get("temporal", {})

    def citations(ids):
        output = []
        for pid in dict.fromkeys(ids):
            p = passages.get(pid)
            if not p:
                continue
            d = docs[p["version_id"]]
            output.append(
                {
                    "passage_id": pid,
                    "version_id": p["version_id"],
                    "title": d["title"],
                    "url": safe_url(d["url"]),
                    "text": p["text"],
                    "start": p["start"],
                    "end": p["end"],
                }
            )
        return output

    timeline = []
    # Prefer the report's explicit dated table; do not invent events from article dates.
    for line in report.split("## Evidence index")[0].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.strip().startswith("|") and len(cells) >= 2 and re.search(r"\d{4}[-年/]", cells[0]):
            timeline.append(
                {
                    "date": cells[0],
                    "text": cells[1],
                    "citations": citations(PIDS.findall(line)),
                    "provenance": "report_table",
                    "verification": "unverified",
                }
            )
    if not timeline and not (run_dir / "answer_state.json").exists():
        for claim in claims:
            if claim.get("event_time"):
                assessment = temporal.get(claim["claim_id"], {})
                timeline.append(
                    {
                        "date": claim["event_time"],
                        "text": claim["text"],
                        "stage": claim.get("stage"),
                        "kind": assessment.get("kind", "unknown"),
                        "citations": citations(claim["passage_ids"]),
                        "provenance": "recorded_claim",
                        "verification": "unverified",
                    }
                )
    timeline.sort(key=lambda row: row["date"])
    sources = []
    for vid, doc in docs.items():
        sources.append(
            {
                "version_id": vid,
                "title": doc["title"],
                "url": safe_url(doc["url"]),
                "content_sha256": evidence.get("document_hashes", {}).get(vid),
                "retrieved_at": doc.get("retrieved_at"),
                "published_at": doc.get("published_at"),
                "available_at": doc.get("available_at"),
                "availability_proof": doc.get("availability_proof"),
                "body_saved": bool(doc.get("content")),
                "delivered_passage_ids": [
                    pid
                    for pid in budget.get("delivered_passages", [])
                    if passages.get(pid, {}).get("version_id") == vid
                ],
                "passages": citations(
                    p["passage_id"] for p in passages.values() if p["version_id"] == vid
                ),
            }
        )
    question_state = maybe_json(run_dir / "question_state.json")
    question_gaps = [
        {
            "gap_id": "question:" + q["question_id"],
            "kind": "question_evidence",
            "status": q["gap_status"],
            "reason": q["text"],
            "owner": "question_state",
            "verification": "model_assessed_unverified",
        }
        for q in question_state.get("questions", {}).values()
        if q["gap_status"] != "addressed_model_assessed"
    ]
    raw_graph = maybe_json(run_dir / "event_graph.json")
    event_view = event_projection(
        raw_graph, docs, events, maybe_json(run_dir / "temporal_review.json")
    )
    adaptive = maybe_json(run_dir / "adaptive_investigation.json")
    if raw_graph:
        timeline = [
            {
                "date": u["finding"]["event_time"].get("start") or "日期未知",
                "date_known": u["finding"]["event_time"]["precision"] != "unknown",
                "date_label": u["time_labels"]["event"],
                "text": u["finding"]["assertion"],
                "kind": u["finding"]["occurrence"],
                "event_unit_id": u["event_unit_id"],
                "citations": citations(r["passage_id"] for r in u["references"]),
                "provenance": "event_graph",
                "verification": "model_assessed_unverified",
            }
            for u in event_view["event_units"]
        ]
        timeline.sort(key=lambda row: (row["date"] == "日期未知", row["date"]))
    snapshot = {
        "workflow": config.get("workflow", "legacy"),
        "event_view": event_view,
        "adaptive": adaptive,
        "run_id": run_dir.name,
        "question": question,
        "task": task,
        "condition": config.get("condition", request.get("condition", "baseline")),
        "adapter": config.get("adapter"),
        "mock": config.get("mock", request.get("mode") == "fixture"),
        "parent_run_id": request.get("parent_run_id"),
        "status": "running" if active else result.get("status", worker_status or "interrupted"),
        "stop_reason": result.get("stop_reason"),
        "model_finalized": result.get("model_finalized"),
        "implementation_hash": config.get("implementation_hash"),
        "model": config.get("model"),
        "model_routing": config.get("model_routing_policy"),
        "model_retries": list(retry_nodes.values()),
        "model_routes": [
            e for e in journal(run_dir / "trajectory.jsonl") if e["kind"] == "model_route"
        ],
        "budget": budget,
        "limits": config.get("budget", {}),
        "nodes": list(nodes.values()),
        "edges": edges,
        "last_sequence": events[-1]["sequence"] if events else 0,
        "network_requests": network_requests,
        "sources": sources,
        "timeline": timeline,
        "claims": claims,
        "gaps": list(investigation.get("gaps", {}).values())
        + question_gaps
        + list(adaptive.get("gaps", {}).values()),
        "report": report.split("## Evidence index")[0].strip(),
        "report_revision": result.get("report_revision"),
        "question_state": question_state,
        "event_graph": raw_graph,
        "temporal_repairs": maybe_json(run_dir / "temporal_repair.json"),
        "repair_execution": maybe_json(run_dir / "repair_execution.json"),
        "trace_notice": "实线为显式计划关联，虚线只表示动作顺序；旧记录不补造计划或决策理由。",
        "timeline_notice": (
            "新流程日期来自事件图原文绑定；未知日期保留。旧流程来自保存报告/主张。均不等于事实认证。"
        ),
    }
    latest_review = maybe_json(run_dir / "temporal_review.json")
    snapshot = apply_temporal_review(snapshot, latest_review)
    if adaptive.get("current_check") and not active and report:
        draft = maybe_json(run_dir / "current_report_draft.json")
        brief = current_brief(adaptive["current_check"], docs, passages, draft)
        snapshot["report_brief"] = brief
        snapshot["report_brief_html"] = render_brief(brief)
    elif config.get("workflow") == "adaptive-event-map" and not active and report:
        review = snapshot.get("temporal_review", {})
        inherited_review = not review.get("decisions") and bool(latest_review)
        if inherited_review:
            review = next(
                (
                    row
                    for path in sorted(
                        run_dir.glob("temporal_reviews/*/record.json"),
                        key=lambda item: item.stat().st_mtime,
                        reverse=True,
                    )
                    if verified_review(row := maybe_json(path), raw_graph)
                ),
                {},
            )
            if review:
                snapshot["temporal_review_latest_status"] = latest_review.get("status")
                snapshot = apply_temporal_review(snapshot, review)
        brief = report_brief(
            snapshot, review, repeat_review_incomplete=inherited_review and bool(review)
        )
        if brief:
            snapshot["report_brief"] = brief
            snapshot["report_brief_html"] = render_brief(brief)
    return snapshot


def apply_temporal_review(snapshot, review):
    """Apply a separate saved audit to presentation, never mutate source graph findings."""
    from .event_report import resolved_temporal_candidate
    from .time_display import span_label

    if not review:
        return snapshot
    valid = verified_review(review, snapshot["event_graph"])
    decisions = {d["finding_id"]: d for d in review["decisions"]} if valid else {}
    snapshot["temporal_review"] = review if valid else {"status": "not_reviewed", "decisions": []}
    resolution_plan = {
        "temporal_review": snapshot["temporal_review"],
        "temporal_repairs": list(snapshot.get("temporal_repairs", {}).get("targets", {}).values()),
        "graph_choices": [
            {
                "finding_id": u["event_unit_id"],
                **u["finding"],
                "passage_ids": [r["passage_id"] for r in u["references"]],
            }
            for u in snapshot.get("event_view", {}).get("event_units", [])
        ],
    }
    for row in snapshot["timeline"]:
        d = decisions.get(row.get("event_unit_id"), {})
        row["temporal_verdict"] = d.get("event_verdict", "not_reviewed")
        if any(not r["same_event_or_claim"] for r in d.get("repairs", [])):
            row["temporal_verdict"] = "target_relation_unconfirmed"
    for unit in snapshot.get("event_view", {}).get("event_units", []):
        d = decisions.get(unit["event_unit_id"])
        unit["temporal_review"] = d
        unit["temporal_repairs"] = [
            t
            for t in snapshot.get("temporal_repairs", {}).get("targets", {}).values()
            if t["finding_id"] == unit["event_unit_id"]
        ]
        if d:
            if (
                d["event_verdict"] != "supported"
                and unit["finding"]["event_time"]["precision"] != "unknown"
            ):
                unit["time_labels"]["event"] = (
                    span_label(unit["finding"]["event_time"]) + "（原记录日期；事件关联未通过审核）"
                )
            unit["time_labels"]["applicability_note"] = {
                "not_applicable": "此条无需单独适用时段（模型审核）。",
                "needs_evidence": "适用范围尚需确认。",
                "supported": "适用时段有对应依据（模型审核）。",
            }[d["applicability"]] + d["applicability_reason"]
        if resolved := resolved_temporal_candidate(
            resolution_plan, unit["event_unit_id"], "event_date_basis"
        ):
            candidate, _ = resolved
            unit["time_labels"]["event"] = (
                span_label(candidate["event_time"]) + "（补充证据已澄清；原记录未改写）"
            )
        if resolved := resolved_temporal_candidate(
            resolution_plan, unit["event_unit_id"], "applicability_scope"
        ):
            candidate, decision = resolved
            unit["time_labels"]["applicability_note"] = (
                "补充发现复审：此条按一次性事件表述；不意味着相关状态永久有效。"
                if decision["applicability"] == "not_applicable"
                else "补充发现复审支持的适用时段：" + span_label(candidate["valid_time"])
            )
    return snapshot


def linked_text(text):
    escaped = html.escape(str(text))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\b(ef-[a-f0-9]{16})\b", r'<a href="#\1">地图节点 ↗</a>', escaped)
    return re.sub(r"\[(p-[a-f0-9]{16})\]", r'<a href="#\1" class="citation">[证据]</a>', escaped)


def report_html(snapshot):
    """Standalone offline HTML: no model request, external scripts or remote assets."""
    if brief := snapshot.get("report_brief"):
        cited = {pid for s in brief["sources"] for pid in s["cited"]}
        excerpts = "".join(
            f'<blockquote id="{html.escape(p["passage_id"], quote=True)}">'
            f"<strong>{html.escape(source['title'])}</strong><p>{html.escape(p['text'])}</p>"
            "</blockquote>"
            for source in snapshot["sources"]
            for p in source["passages"]
            if p["passage_id"] in cited
        )
        styles = Path(__file__).with_name("web_assets").joinpath("report.css").read_text()
        return (
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>研究报告 · Trace Atlas</title><style>"
            + styles
            + '</style><main class="brief-export"><h1>'
            + html.escape(snapshot["question"])
            + "</h1>"
            + ("<p>脚本模拟验收 · 合成资料</p>" if snapshot["mock"] else "")
            + ("<p>部分完成</p>" if snapshot["status"] == "partial" else "")
            + render_brief(brief, standalone=True)
            + "<details><summary>保存的引用原文</summary>"
            + excerpts
            + "</details></main></html>"
        )
    lines = []
    table = False
    for line in snapshot["report"].splitlines():
        if line.startswith("|"):
            if not table:
                lines.append("<table>")
                table = True
            cells = line.strip("| ").split("|")
            if not all(re.fullmatch(r"[\s:\-]+", cell) for cell in cells):
                lines.append(
                    "<tr>" + "".join("<td>" + linked_text(c) + "</td>" for c in cells) + "</tr>"
                )
            continue
        if table:
            lines.append("</table>")
            table = False
        heading = re.match(r"^(#{1,4}) (.*)", line)
        if heading:
            lines.append(f"<h3>{linked_text(heading[2])}</h3>")
        elif line.strip():
            lines.append("<p>" + linked_text(line) + "</p>")
    if table:
        lines.append("</table>")

    def timeline_cards(rows):
        return "".join(
            '<article class="milestone"'
            + (
                ' id="' + html.escape(row["event_unit_id"], quote=True) + '"'
                if row.get("event_unit_id")
                else ""
            )
            + "><time>"
            + html.escape(str(row.get("date_label", row["date"])))
            + "</time><p>"
            + linked_text(row["text"])
            + "</p>"
            + "".join(
                '<a href="#' + c["passage_id"] + '">查看原文证据 ↗</a> ' for c in row["citations"]
            )
            + "</article>"
            for row in rows
        )

    dated = [
        r
        for r in snapshot["timeline"]
        if r.get("date_known", r["date"] != "日期未知")
        and r.get("temporal_verdict") in {None, "supported"}
    ]
    undated = [r for r in snapshot["timeline"] if r not in dated]
    timeline = timeline_cards(dated) or "<p>尚未形成带日期依据的时间节点。</p>"
    if undated:
        timeline += (
            "<details><summary>日期待确认与待复核的相关证据（"
            + str(len(undated))
            + " 条）</summary><p>这些发现保留原文和链接，暂不放入有序时间线。</p>"
            + timeline_cards(undated)
            + "</details>"
        )
    source_sections = []
    for source in snapshot["sources"]:
        link = (
            '<a target="_blank" rel="noopener noreferrer" href="'
            + html.escape(source["url"], quote=True)
            + '">打开来源 ↗</a>'
            if source["url"]
            else "来源网址不可用"
        )
        cited = set(PIDS.findall(snapshot["report"])) | {
            c["passage_id"] for row in snapshot["timeline"] for c in row["citations"]
        }
        passages = "".join(
            '<blockquote id="'
            + p["passage_id"]
            + '"><small>'
            + p["passage_id"]
            + f" · 原文位置 {p['start']}–{p['end']}</small><p>"
            + html.escape(p["text"])
            + "</p></blockquote>"
            for p in source["passages"]
            if p["passage_id"] in cited
        )
        source_sections.append(
            "<article><h3>"
            + html.escape(source["title"])
            + "</h3>"
            + link
            + passages
            + "</article>"
        )
    graph = "".join(
        "<li><small>"
        + html.escape(node.get("label", node["kind"]))
        + "</small><strong>"
        + html.escape(node["title"])
        + "</strong></li>"
        for node in snapshot["nodes"]
        if node["id"] != "task"
    )
    b = snapshot["budget"]
    status = html.escape(snapshot["status"] + " / " + (snapshot["stop_reason"] or "agent_finished"))
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>研究报告 · Trace Atlas</title>
<style>
body{margin:0;background:#f5f4ee;color:#24352e;font:15px/1.8 system-ui,sans-serif}
main{max-width:1000px;margin:auto;padding:48px 32px}h1{font-size:32px;line-height:1.4}
h2{margin-top:42px;border-top:1px solid #d4dbce;padding-top:24px}small,time{color:#647468}
a{color:#236b56}article{background:white;border:1px solid #dce1d5;padding:20px;margin:14px 0;
border-radius:12px}.milestone{border-left:4px solid #719479;margin-left:12px;position:relative}
.milestone:before{content:'';position:absolute;left:-13px;top:25px;border-radius:50%;width:16px;
height:16px;background:#295e4c}.metrics{display:flex;gap:32px;flex-wrap:wrap}
blockquote{border-left:2px solid #a4baab;margin:20px 0;padding:4px 16px;white-space:pre-wrap}
td{padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse;width:100%}
.trace{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;list-style:none;padding:0}
.trace li{background:#e9eee4;padding:12px;border-radius:8px}
.trace strong{display:block;font-size:13px}
@media(max-width:600px){main{padding:24px 16px}.trace{grid-template-columns:1fr}}
@media print{body{background:white}article,li{break-inside:avoid}a{color:inherit}}
</style><main><small>TRACE ATLAS / EVIDENCE-GROUNDED RESEARCH</small>""" + (
        "<h1>"
        + html.escape(snapshot["question"])
        + "</h1><p>"
        + status
        + "</p>"
        + ("<p>脚本模拟验收 · 合成资料，不是真实新闻调查</p>" if snapshot["mock"] else "")
        + '<div class="metrics"><span>模型请求 '
        + str(b.get("model_calls", 0))
        + "</span><span>Token "
        + format(b.get("tokens", 0), ",")
        + "</span><span>网页请求 "
        + str(b.get("retrieval_api_calls", 0))
        + "</span></div>"
        + "<h2>带证据的事件时间线</h2><p>"
        + snapshot["timeline_notice"]
        + "</p><p>"
        + DATE_NOTE
        + "</p>"
        + timeline
        + "<h2>研究报告</h2>"
        + "\n".join(lines)
        + "<h2>调查动作记录</h2>"
        + "<p>按实际调用顺序导出；不推断动作间的因果关系。</p>"
        + '<ol class="trace">'
        + graph
        + "</ol>"
        + "<h2>引用原文与来源</h2>"
        + "".join(source_sections)
        + "<footer><p>Run: "
        + html.escape(snapshot["run_id"])
        + " · 模型判断和引用存在性均不等于事实核验通过。"
        + "实时网页不是已验证的历史快照。</p></footer></main></html>"
    )
