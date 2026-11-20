"""First failures require an actionable correction before retrying."""

import asyncio
import json

from test_adaptive import fixture, gap


def invoke(tool, **args):
    return json.loads(asyncio.run(tool.ainvoke(args)))


def search_args(gap_id):
    return {
        "gap_id": gap_id,
        "query": "current policy status",
        "strategy": "primary_record",
        "reason": "Find a decisive public source",
        "window": {"precision": "unknown"},
        "movement": "unbounded",
    }


def test_first_empty_round_pauses_search_and_creates_independent_child_tasks(tmp_path):
    _, b, g, a = fixture(tmp_path)
    gid = gap(a, "question_or_explanation_unresolved")
    tools = {t.name: t for t in a.tools()}
    args = search_args(gid)
    assert not invoke(tools["search_gap"], **args)["results"]
    prompt = json.loads(a.prompt().split("\n", 1)[1])
    assert prompt["review_failed_attempt"]["failure"] == "no_candidates"
    assert [t.name for t in a.surface(list(tools.values()))] == ["replan_gap"]
    assert "replan_gap" in invoke(tools["search_gap"], **args)["error"]
    assert len(a.search.calls) == 1
    children = invoke(
        tools["replan_gap"],
        gap_id=gid,
        route="split",
        reason="Check phases independently",
        questions=["Which agency issued the first decision?", "When did the policy take effect?"],
    )["children"]
    assert len(children) == 2 and gid not in children
    assert g.node_view("premise")["question_coverage"] == "unresolved"
    child_args = {**args, "gap_id": children[0], "query": "first decision from agency"}
    invoke(tools["search_gap"], **child_args)
    a.prompt()
    assert a.data["gaps"][children[1]]["status"] == "open"
    assert b.state["model_calls"] == 0


def test_source_failure_points_to_other_candidate_and_rejects_cosmetic_retry(tmp_path):
    _, b, _, a = fixture(tmp_path)
    gid = gap(a, "question_or_explanation_unresolved")
    calls = []

    class Candidates:
        async def ainvoke(self, args, config):
            calls.append(args)
            return '{"results":[{"version_id":"failed"},{"version_id":"alternative"}]}'

    a.search = Candidates()
    tools = {t.name: t for t in a.tools()}
    args = search_args(gid)
    invoke(tools["search_gap"], **args)
    read = {"name": "read_document", "id": "read-failed", "args": {"version_id": "failed"}}
    assert a.before_tool(read) is None
    a.after_tool(read, '{"error":"page_extraction_failed"}')
    assert a.runtime_summary()["active_round"]["source_feedback"]["other_candidates"] == [
        "alternative"
    ]
    assert "another candidate" in a.before_tool({**read, "id": "repeat"})
    a.close_round("No usable original retrieved")
    decision = {
        "gap_id": gid,
        "route": "change_route",
        "reason": "Use independent reporting",
    }
    assert "source_domains" in invoke(tools["replan_gap"], **decision)["error"]
    assert len(calls) == 1
    assert (
        invoke(
            tools["replan_gap"],
            **decision,
            source_domains=["archives.example"],
        )["failure"]
        == "original_not_obtained"
    )
    invoke(tools["search_gap"], **args)
    assert calls[-1]["include_domains"] == ["archives.example"]
    assert len(calls) == 2 and b.state["model_calls"] == 0
