import asyncio
import json

import pytest
from pydantic import ValidationError

from news_agent.reporting import validate_and_append_sources
from news_agent.retrieval import LocalCorpus
from news_agent.schemas import ResearchTask
from news_agent.storage import EvidenceStore, RunStore


def test_cutoff_filters_titles_snippets_and_direct_reads(task, corpus_path):
    corpus = LocalCorpus.load(corpus_path, task)
    output = json.dumps(corpus.search("Aurora Birch acquisition completed", 10))
    assert "closing-after-cutoff" not in output
    assert "old-url-new-version" not in output
    assert "FUTURE_COMPLETION_SENTINEL" not in output
    assert len(corpus.documents) == 3
    for version_id in ["closing-after-cutoff", "old-url-new-version", "nonexistent"]:
        with pytest.raises(ValueError, match="not available"):
            corpus.read(version_id)


def test_unknown_availability_does_not_pass_asof(task, corpus_path):
    document = LocalCorpus.load(corpus_path, task).read("announcement-v1")
    unknown = document.model_copy(update={"availability_proof": None})
    assert not LocalCorpus([unknown], task).search("Aurora")
    retrospective = task.model_copy(update={"mode": "retrospective", "cutoff": None})
    assert LocalCorpus([unknown], retrospective).search("Aurora")


def test_cutoff_requires_timezone_and_value():
    with pytest.raises(ValidationError):
        ResearchTask(task_id="x", event="e", question="q", mode="as_of")
    with pytest.raises(ValidationError):
        ResearchTask(task_id="x", event="e", question="q", mode="as_of", cutoff="2026-06-10")


def test_offsets_immutable_version_and_unverified_claims(tmp_path, task, corpus_path):
    evidence = EvidenceStore(RunStore(tmp_path))
    doc = LocalCorpus.load(corpus_path, task).read("announcement-v1")
    doc = doc.model_copy(update={"content": "First paragraph.\n\nSecond paragraph."})
    passages = evidence.add_document(doc)
    for passage in passages:
        assert doc.content[passage.start : passage.end] == passage.text
    claim = evidence.add_claim("An announcement exists.", [passages[0].passage_id], None, None)
    assert claim.verification == "unverified"
    assert EvidenceStore(RunStore(tmp_path)).claims[0] == claim
    with pytest.raises(ValueError, match="immutable"):
        evidence.add_document(doc.model_copy(update={"content": "changed"}))
    with pytest.raises(ValueError, match="already read"):
        evidence.add_claim("fabricated", ["p-" + "f" * 16], None, None)
    text, invalid = validate_and_append_sources("[p-" + "f" * 16 + "]", evidence)
    assert invalid and "INVALID CITATION REMOVED" in text
    saved = json.loads((tmp_path / "evidence.json").read_text())
    saved["documents"][doc.version_id]["content"] = "tampered after the checkpoint"
    (tmp_path / "evidence.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="hash mismatch"):
        EvidenceStore(RunStore(tmp_path))


def test_long_article_is_paginated_with_original_offsets(tmp_path, task, corpus_path):
    from news_agent.budget import Budget
    from news_agent.harness import make_tools
    from news_agent.schemas import BudgetConfig

    doc = LocalCorpus.load(corpus_path, task).read("announcement-v1")
    doc = doc.model_copy(update={"content": "Long original evidence. " * 400})
    store = RunStore(tmp_path)
    evidence = EvidenceStore(store)
    tools = make_tools(LocalCorpus([doc], task), evidence, Budget(BudgetConfig(), store))
    read = next(t for t in tools if t.name == "read_document")
    first = json.loads(
        asyncio.run(
            read.ainvoke(
                {
                    "version_id": doc.version_id,
                    "passage_limit": 1,
                }
            )
        )
    )
    assert first["next_offset"] == 1
    assert len(first["passages"][0]["text"]) <= 1500
    second = json.loads(
        asyncio.run(
            read.ainvoke(
                {
                    "version_id": doc.version_id,
                    "passage_limit": 1,
                    "passage_offset": 1,
                }
            )
        )
    )
    assert second["passages"][0]["passage_id"] != first["passages"][0]["passage_id"]
    for passage in evidence.passages.values():
        assert doc.content[passage.start : passage.end] == passage.text
