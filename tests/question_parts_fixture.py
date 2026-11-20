"""Synthetic question parts used by the adaptive and web workspace tests."""

from test_event_graph import setup

from news_agent.adaptive import AdaptiveInvestigation
from news_agent.event_graph import Finding, QuestionPart


def prepared(root):
    task, e, b, g = setup(root)
    e.event_graph, e.adaptive = g, AdaptiveInvestigation(g, b)
    parts = [
        QuestionPart(part_id=k, question=q, required_occurrence=t)
        for k, q, t in [
            ("launch", "发射是否完成？", "observed"),
            ("plan", "任务计划是什么？", "planned"),
            ("orbit", "入轨是否完成？", "observed"),
        ]
    ]
    g.record(
        "q",
        "事件发展",
        "Probe",
        "question",
        "major",
        "用户的三个问题",
        Finding(
            assertion="三个问题待查", stance="unknown", occurrence="unknown", uncertainty="尚无原文"
        ),
        0,
        parts,
    )
    return task, e, b, g


def bind(g, ref, part, occurrence, text, stance="supports"):
    return g.append(
        "q",
        Finding(
            assertion=text,
            stance=stance,
            occurrence=occurrence,
            originals=[ref],
            question_part_id=part,
        ),
        len(g.data["nodes"]["q"]["findings"]),
    )
