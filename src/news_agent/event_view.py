"""Read-only event projection. Never opens a checkpoint or invokes a model."""

from types import SimpleNamespace

from .event_graph import EventGraph, digest
from .time_display import time_labels


def event_projection(raw, docs, events, review=None):
    if not raw:
        return {"nodes": [], "event_units": [], "relations": [], "available": False}
    if raw.get("sha256") != digest({k: v for k, v in raw.items() if k != "sha256"}):
        return {
            "nodes": [],
            "event_units": [],
            "relations": [],
            "available": False,
            "error": "事件记录校验未通过，暂不展示推导状态",
        }
    graph = EventGraph.__new__(EventGraph)
    graph.data = raw
    graph.evidence = SimpleNamespace(documents=docs, source_temporal_review=review)
    try:
        view = graph.snapshot()
    except (KeyError, ValueError):
        return {
            "nodes": [],
            "event_units": [],
            "relations": [],
            "available": False,
            "error": "证据与事件记录尚未同步，请等待下一次刷新",
        }
    commits = {
        e.get("finding_id"): e
        for e in events
        if e["kind"] == "event_graph_changed" and e.get("finding_id")
    }
    for unit in view["event_units"]:
        saved = next(
            f
            for n in raw["nodes"].values()
            for f in n["findings"]
            if f["finding_id"] == unit["event_unit_id"]
        )
        unit["source_context"] = saved.get("source_context", [])
        unit["time_labels"] = time_labels(
            unit["finding"]["event_time"], unit["finding"]["valid_time"]
        )
        commit = commits.get(unit["event_unit_id"])
        unit["sequence"] = commit["sequence"] if commit else None
        unit["action_id"] = commit.get("action_id") if commit else None
    view["available"] = True
    view["notice"] = "事件状态来自保存的原文绑定与模型评估，不是独立事实认证。"
    return view
