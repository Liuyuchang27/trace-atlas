import asyncio
import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from news_agent.budget import Budget, BudgetCallback
from news_agent.context import EvidenceContext
from news_agent.retrieval import LocalCorpus
from news_agent.revision import revise_report
from news_agent.schemas import BudgetConfig
from news_agent.storage import EvidenceStore, RunStore, read_json
from news_agent.temporal_review import fixture_review


class RevisionModel(BaseChatModel):
    reply: str
    finish: str = "stop"
    requests: int = 0

    @property
    def _llm_type(self):
        return "revision-transport-fixture"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests += 1
        assert messages[0].text.startswith(("REPORT_REVISION:", "TEMPORAL_REVIEW:"))
        assert not kwargs.get("tools")
        payload = json.loads(messages[-1].text)
        assert "FUTURE_COMPLETION_SENTINEL" not in json.dumps(payload)
        if "output_schema" not in payload:
            assert payload["assessment_constraints"] == {"assessments": [], "omitted": 0}
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=(
                            json.dumps(fixture_review(payload))
                            if messages[0].text.startswith("TEMPORAL_REVIEW:")
                            else self.reply
                        ),
                        response_metadata={"finish_reason": self.finish},
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        },
                    )
                )
            ]
        )


def setup(tmp_path, task, corpus_path):
    store = RunStore(tmp_path)
    evidence = EvidenceStore(store)
    for doc in LocalCorpus.load(corpus_path, task).documents.values():
        evidence.add_document(doc)
    budget = Budget(BudgetConfig(), store)
    context = EvidenceContext(RevisionModel(reply=""), task, evidence, budget)
    pid = next(iter(evidence.passages))
    return budget, context, pid, {"callbacks": [BudgetCallback(budget, 0)]}


@pytest.mark.parametrize("bad", [None, "truncated", "invalid_citation", "uncited"])
def test_revision_is_one_metered_request_with_draft_fallback_and_cached_resume(
    tmp_path, task, corpus_path, bad
):
    budget, context, pid, config = setup(tmp_path, task, corpus_path)
    draft = f"Draft status [{pid}]."
    reply = f"Revised status limited to supplied evidence [{pid}]."
    if bad == "invalid_citation":
        reply = "Invented [p-0000000000000000]."
    if bad == "uncited":
        reply = "No references."
    model = RevisionModel(reply=reply, finish="length" if bad == "truncated" else "stop")
    result, record = asyncio.run(revise_report(draft, task, context, None, model, budget, config))
    assert record["status"] == ("applied" if bad is None else "retained_draft")
    assert result == (reply if bad is None else draft)
    assert model.requests == budget.state["model_calls"] == 1
    assert budget.state["tokens"] == 120
    assert (tmp_path / "report_draft.md").read_text() == draft
    assert (tmp_path / "report_revision_output.md").read_text() == reply
    again, _ = asyncio.run(revise_report(draft, task, context, None, model, budget, config))
    assert again == result and model.requests == 1
    events = [json.loads(s) for s in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert next(e for e in events if e["kind"] == "model_dispatch")["phase"] == "repair"
    assert budget.state["phase"] == "research"


def test_revision_refuses_over_budget_and_never_retries_interrupted_request(
    tmp_path, task, corpus_path
):
    budget, context, pid, config = setup(tmp_path, task, corpus_path)
    budget.state["tokens"] = 79900
    model = RevisionModel(reply=f"Answer [{pid}].")
    draft = f"Draft [{pid}]."
    result, record = asyncio.run(revise_report(draft, task, context, None, model, budget, config))
    assert result == draft and record["reason"] == "token_limit"
    assert budget.state["model_calls"] == model.requests == 0
    # Simulate a crash after write-ahead intent, before any complete response.
    saved = read_json(tmp_path / "report_revision.json")
    saved["status"] = "requested"
    (tmp_path / "report_revision.json").write_text(json.dumps(saved))
    budget.state["tokens"] = 0
    asyncio.run(revise_report(draft, task, context, None, model, budget, config))
    assert model.requests == 0


def test_revision_cache_tampering_retains_draft_without_rebilling(tmp_path, task, corpus_path):
    budget, context, pid, config = setup(tmp_path, task, corpus_path)
    draft = f"Draft [{pid}]."
    model = RevisionModel(reply=f"Revision [{pid}].")
    asyncio.run(revise_report(draft, task, context, None, model, budget, config))
    (tmp_path / "report_revision_output.md").write_text("Replaced cache text.")
    result, record = asyncio.run(revise_report(draft, task, context, None, model, budget, config))
    assert result == draft and record["status"] == "retained_draft"
    assert model.requests == 1
