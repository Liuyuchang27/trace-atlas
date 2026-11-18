import asyncio
import json

import httpx
import pytest

from news_agent.budget import Budget
from news_agent.retrieval import RetrievalFailure, TavilyCorpus
from news_agent.schemas import BudgetConfig
from news_agent.storage import RunStore, read_json


def mock_http(monkeypatch, respond):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(respond), **kw)
    )


def test_search_discovers_candidates_and_only_selected_read_fetches_body(
    task, monkeypatch, tmp_path
):
    calls = []

    def respond(request):
        calls.append((request.url.path, json.loads(request.content)))
        assert request.headers["authorization"] == "Bearer test-not-a-real-key"
        body = json.loads(request.content)
        if request.url.path == "/search":
            assert body["include_raw_content"] is False and body["include_answer"] is False
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "First",
                            "url": "https://source.example/first",
                            "content": "Discovery only.",
                        },
                        {
                            "title": "Second",
                            "url": "https://source.example/second",
                            "content": "Other discovery.",
                        },
                    ],
                    "usage": {"credits": 1},
                },
            )
        assert body["urls"] == ["https://source.example/second"]
        assert "query" not in body  # don't replace the original page with reranked snippets
        return httpx.Response(
            200,
            json={
                "results": [{"url": body["urls"][0], "raw_content": "Original selected text."}],
                "usage": {"credits": 0.2},
            },
        )

    mock_http(monkeypatch, respond)
    retro = task.model_copy(update={"mode": "retrospective", "cutoff": None})
    store = RunStore(tmp_path)
    budget = Budget(BudgetConfig(), store)
    corpus = TavilyCorpus(retro, "test-not-a-real-key", store, budget)
    hits = asyncio.run(corpus.asearch("agent chosen query"))
    assert not corpus.documents and len(calls) == 1
    with pytest.raises(ValueError, match="not been extracted"):
        corpus.read(hits[1]["version_id"])
    doc = asyncio.run(corpus.aread(hits[1]["version_id"]))
    assert doc.content == "Original selected text." and doc.available_at is None
    assert doc.retrieved_at is not None and len(calls) == 2
    assert hits[0]["version_id"] not in corpus.documents
    restored = TavilyCorpus(retro, "test-not-a-real-key", store, budget)
    assert asyncio.run(restored.aread(hits[1]["version_id"])) == doc and len(calls) == 2
    assert budget.state["retrieval_api_calls"] == 2
    assert budget.state["retrieval_credits"] == 1.2 and not budget.state["retrieval_pending"]


def test_live_search_does_not_claim_strict_historical_availability(task):
    with pytest.raises(ValueError, match="retrospective only"):
        TavilyCorpus(task, "test-not-a-real-key")


def test_model_chosen_domain_filter_is_forwarded_and_invalid_input_does_not_dispatch(
    task, monkeypatch
):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["include_domains"] == ["issuer.example", "regulator.example"]
        assert body["include_answer"] is False and body["include_raw_content"] is False
        return httpx.Response(200, json={"results": [], "usage": {"credits": 1}})

    mock_http(monkeypatch, respond)
    corpus = TavilyCorpus(task.model_copy(update={"mode": "retrospective"}), "test-key")
    asyncio.run(
        corpus.asearch("agent query", include_domains=["issuer.example", "regulator.example"])
    )
    for domains in [["https://issuer.example/path"], ["issuer.example"] * 9]:
        with pytest.raises(RetrievalFailure, match="include_domains"):
            asyncio.run(corpus.asearch("agent query", include_domains=domains))
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["http", "extract", "empty", "url_mismatch"])
def test_failed_extraction_never_turns_search_snippet_into_evidence(task, monkeypatch, failure):
    def respond(request):
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Candidate",
                            "url": "https://source.example/one",
                            "content": "NOT ORIGINAL EVIDENCE",
                        }
                    ]
                },
            )
        if failure == "http":
            return httpx.Response(429, json={"detail": "provider details"})
        if failure == "extract":
            return httpx.Response(
                200, json={"results": [], "failed_results": [{"error": "unavailable"}]}
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://other.example/"
                        if failure == "url_mismatch"
                        else "https://source.example/one",
                        "raw_content": "" if failure == "empty" else "Text",
                    }
                ]
            },
        )

    mock_http(monkeypatch, respond)
    corpus = TavilyCorpus(task.model_copy(update={"mode": "retrospective"}), "test-not-a-real-key")
    hits = asyncio.run(corpus.asearch("query"))
    with pytest.raises(RetrievalFailure):
        asyncio.run(corpus.aread(hits[0]["version_id"]))
    assert not corpus.documents


def test_credential_redaction_and_unknown_provider_usage(task, monkeypatch, tmp_path):
    secret = "test-key-not-for-logging"

    def respond(request):
        return httpx.Response(200, json={"results": [], "debug": secret})

    mock_http(monkeypatch, respond)
    store = RunStore(tmp_path)
    budget = Budget(BudgetConfig(), store)
    corpus = TavilyCorpus(task.model_copy(update={"mode": "retrospective"}), secret, store, budget)
    asyncio.run(corpus.asearch("query"))
    assert budget.state["retrieval_unknown_usage_calls"] == 1
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert secret.encode() not in p.read_bytes()


def test_open_url_validation_and_extraction_truncation(task, monkeypatch):
    from news_agent.retrieval import public_url

    for url in [
        "file:///etc/passwd",
        "https://localhost/",
        "http://127.0.0.1/",
        "http://169.254.169.254/",
        "https://name:secret@example.com/",
    ]:
        with pytest.raises(ValueError):
            public_url(url)

    def respond(request):
        return httpx.Response(
            200,
            json={"results": [{"url": "https://source.example/one", "raw_content": "A" * 65000}]},
        )

    mock_http(monkeypatch, respond)
    corpus = TavilyCorpus(task.model_copy(update={"mode": "retrospective"}), "test-not-a-real-key")
    vid = corpus.register("https://source.example/one#part")
    doc = asyncio.run(corpus.aread(vid))
    assert len(doc.content) == 60000 and corpus.source_metadata[vid]["truncated"]


def test_interrupted_provider_request_is_preserved_as_unknown_on_resume(task, tmp_path):
    store = RunStore(tmp_path)
    budget = Budget(BudgetConfig(), store)
    budget.state.update(retrieval_api_calls=1, retrieval_pending={"old-request": "search"})
    budget.save()
    TavilyCorpus(
        task.model_copy(update={"mode": "retrospective"}), "test-not-a-real-key", store, budget
    )
    assert read_json(tmp_path / "budget.json")["retrieval_unknown_usage_calls"] == 1
    assert not budget.state["retrieval_pending"]
