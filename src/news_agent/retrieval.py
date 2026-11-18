"""All adapters filter before returning titles, snippets, IDs or content to the model."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .schemas import DocumentVersion, ResearchTask
from .storage import read_json


def eligible(document: DocumentVersion, task: ResearchTask) -> bool:
    if task.mode == "retrospective":
        return True
    return bool(
        document.available_at is not None
        and document.availability_proof
        and document.availability_proof.strip()
        and task.cutoff is not None
        and document.available_at <= task.cutoff
    )


class LocalCorpus:
    def __init__(self, documents: list[DocumentVersion], task: ResearchTask):
        ids = [d.version_id for d in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("corpus contains duplicate version IDs")
        self.task = task
        self.documents = {d.version_id: d for d in documents if eligible(d, task)}

    @classmethod
    def load(cls, path: Path, task: ResearchTask):
        return cls([DocumentVersion.model_validate(d) for d in read_json(path)], task)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for doc in self.documents.values():
            text = (doc.title + " " + doc.content).lower()
            score = sum(len(re.findall(r"\b" + re.escape(term) + r"\b", text)) for term in terms)
            if score:
                scored.append((score, doc.version_id, doc))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [
            {
                "version_id": doc.version_id,
                "title": doc.title,
                "url": doc.url,
                "snippet": doc.content[:260],
                "published_at": str(doc.published_at) if doc.published_at else None,
                "available_at": str(doc.available_at) if doc.available_at else None,
            }
            for _, _, doc in scored[: max(1, min(limit, 10))]
        ]

    def read(self, version_id: str) -> DocumentVersion:
        # An excluded ID is indistinguishable from a nonexistent ID.
        if version_id not in self.documents:
            raise ValueError("document is not available in this run")
        return self.documents[version_id]

    async def asearch(self, query: str, limit: int = 5, timeout: float = 30):
        return self.search(query, limit)


class RetrievalFailure(ValueError):
    """Safe provider failure code, without credential-bearing response bodies."""


def live_retrieval_policy():
    return {
        "version": "tavily-agent-search-extract-v2",
        "search": "basic_candidates_only_no_answer_no_raw_content",
        "read": "agent_selected_url_basic_extract_markdown_no_reranking",
        "version_identity": "url_handle_immutable_first_extraction_with_content_hash",
        "max_document_characters": 60000,
        "snippet_characters": 400,
        "automatic_retries": 0,
        "forced_review_sampling": "up_to_two_selected_pages_not_all_candidates",
        "context_unread_candidate_limit": 6,
        "strict_as_of": False,
        "provider_usage": "credits_when_reported_otherwise_unknown",
    }


def public_url(url):
    from ipaddress import ip_address
    from urllib.parse import urlsplit, urlunsplit

    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        raise ValueError("a public http(s) URL without embedded credentials is required")
    if host == "localhost" or host.endswith((".localhost", ".local")):
        raise ValueError("a public web URL is required")
    try:
        address = ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("a public web URL is required")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


class TavilyCorpus:
    def __init__(self, task: ResearchTask, api_key: str, store=None, budget=None):
        if task.mode != "retrospective":
            raise ValueError(
                "live Tavily is retrospective only; strict as_of requires verified snapshots"
            )
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required; fill it in the project's local .env")
        self.api_key, self.store, self.budget = api_key, store, budget
        self.documents: dict[str, DocumentVersion] = {}
        self.candidates, self.source_metadata = {}, {}
        if store and (store.root / "retrieval_cache.json").exists():
            cached = read_json(store.root / "retrieval_cache.json")
            if cached.get("version") != live_retrieval_policy()["version"]:
                raise ValueError("live retrieval cache policy changed")
            self.candidates = cached["candidates"]
            self.source_metadata = cached["source_metadata"]
            self.documents = {
                k: DocumentVersion.model_validate(v) for k, v in cached["documents"].items()
            }
        if budget:
            budget.state["retrieval_mode"] = "live"
            # A process crash may leave a dispatched request without usage. Do
            # not erase that uncertainty or claim remote exactly-once behavior.
            for rid, operation in list(budget.state.get("retrieval_pending", {}).items()):
                self._settle(rid, operation, None, error="interrupted_unknown_usage")
            budget.save()

    def persist(self):
        if self.store:
            self.store.save(
                "retrieval_cache.json",
                {
                    "version": live_retrieval_policy()["version"],
                    "candidates": self.candidates,
                    "documents": {k: v.model_dump(mode="json") for k, v in self.documents.items()},
                    "source_metadata": self.source_metadata,
                },
            )

    def register(self, url, title="", snippet="", published_date=None):
        import hashlib

        url = public_url(url)
        version_id = "web-" + hashlib.sha256(url.encode()).hexdigest()[:20]
        if version_id not in self.candidates:
            self.candidates[version_id] = {
                "version_id": version_id,
                "title": title or url,
                "url": url,
                "snippet": snippet[: live_retrieval_policy()["snippet_characters"]],
                "provider_published_date": published_date,
            }
        return version_id

    def _settle(self, rid, operation, data, error=None, status_code=None):
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        credits = usage.get("credits") if isinstance(usage, dict) else None
        known = isinstance(credits, (int, float)) and not isinstance(credits, bool) and credits >= 0
        if self.budget:
            s = self.budget.state
            s.setdefault("retrieval_pending", {}).pop(rid, None)
            if known:
                s["retrieval_credits"] = s.get("retrieval_credits", 0) + credits
            else:
                s["retrieval_unknown_usage_calls"] = s.get("retrieval_unknown_usage_calls", 0) + 1
            self.budget.save()
        if self.store:
            self.store.event(
                "retrieval_result",
                request_id=rid,
                operation=operation,
                provider_request_id=data.get("request_id") if isinstance(data, dict) else None,
                credits=credits if known else None,
                usage_known=known,
                status_code=status_code,
                error=error,
            )

    async def _request(self, operation, payload, timeout):
        import json
        from uuid import uuid4

        rid = uuid4().hex
        if self.budget:
            if self.budget.remaining_time() <= 0:
                raise RetrievalFailure("retrieval_deadline_reached")
            timeout = min(timeout, self.budget.remaining_time())
            s = self.budget.state
            s["retrieval_api_calls"] = s.get("retrieval_api_calls", 0) + 1
            key = "web_search_requests" if operation == "search" else "web_extract_requests"
            s[key] = s.get(key, 0) + 1
            s.setdefault("retrieval_pending", {})[rid] = operation
            self.budget.save()
        if self.store:
            self.store.event(
                "retrieval_dispatch", request_id=rid, operation=operation, arguments=payload
            )
        data, error, status = None, None, None
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    "https://api.tavily.com/" + operation,
                    headers={"Authorization": "Bearer " + self.api_key},
                    json=payload,
                )
            status = response.status_code
            if status != 200:
                codes = {
                    401: "tavily_authentication_failed",
                    403: "tavily_access_denied",
                    429: "tavily_rate_limited",
                    432: "tavily_plan_limit",
                    433: "tavily_usage_limit",
                }
                raise RetrievalFailure(codes.get(status, "tavily_http_" + str(status)))
            try:
                data = json.loads(response.text.replace(self.api_key, "[REDACTED]"))
            except ValueError as exc:
                raise RetrievalFailure("tavily_invalid_response") from exc
            if not isinstance(data, dict):
                raise RetrievalFailure("tavily_invalid_response")
            if self.store:
                self.store.save(
                    "web_responses/" + rid + ".json",
                    {
                        "operation": operation,
                        "arguments": payload,
                        "observed_at": datetime.now(UTC).isoformat(),
                        "response": data,
                    },
                )
            return data
        except BaseException as exc:
            error = str(exc) if isinstance(exc, RetrievalFailure) else type(exc).__name__
            raise
        finally:
            self._settle(rid, operation, data, error, status)

    async def asearch(
        self,
        query: str,
        limit: int = 5,
        timeout: float = 30,
        include_domains: list[str] | None = None,
    ) -> list[dict]:
        domains = []
        for domain in include_domains or []:
            if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", domain):
                raise RetrievalFailure("include_domains_requires_hostnames")
            domains.append(domain.lower())
        if len(domains) > 8:
            raise RetrievalFailure("include_domains_limit")
        data = await self._request(
            "search",
            {
                "query": query,
                "max_results": max(1, min(limit, 10)),
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
                "include_usage": True,
                **({"include_domains": domains} if domains else {}),
            },
            timeout,
        )
        results = data.get("results")
        if not isinstance(results, list):
            raise RetrievalFailure("tavily_invalid_search_results")
        hits = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                vid = self.register(
                    item["url"],
                    item.get("title", ""),
                    item.get("content") or "",
                    item.get("published_date"),
                )
            except (ValueError, KeyError):
                continue
            self.candidates[vid]["discovery_query"] = query
            hits.append(dict(self.candidates[vid]))
        self.persist()
        return hits

    async def aread(self, version_id: str, timeout: float = 30) -> DocumentVersion:
        if version_id in self.documents:
            return self.documents[version_id]
        if version_id not in self.candidates:
            raise ValueError("read requires a returned candidate ID; use open_url for a new URL")
        candidate = self.candidates[version_id]
        data = await self._request(
            "extract",
            {
                "urls": [candidate["url"]],
                "extract_depth": "basic",
                "format": "markdown",
                "include_images": False,
                "include_usage": True,
            },
            timeout,
        )
        results = data.get("results", [])
        if not isinstance(results, list) or not results:
            raise RetrievalFailure("page_extraction_failed")
        item = next(
            (r for r in results if isinstance(r, dict) and r.get("url") == candidate["url"]), None
        )
        if item is None:
            raise RetrievalFailure("extraction_url_mismatch")
        text = item.get("raw_content")
        if not isinstance(text, str) or not text.strip():
            raise RetrievalFailure("page_has_no_extracted_content")
        maximum = live_retrieval_policy()["max_document_characters"]
        observed = datetime.now(UTC)
        doc = DocumentVersion(
            version_id=version_id,
            title=item.get("title") or candidate["title"],
            url=candidate["url"],
            content=text[:maximum],
            retrieved_at=observed,
        )
        self.documents[version_id] = doc
        self.source_metadata[version_id] = {
            "extracted_characters": len(text),
            "saved_characters": len(doc.content),
            "truncated": len(text) > maximum,
            "content_sha256": doc.content_hash,
            "observed_at": observed.isoformat(),
            "extraction": "Tavily extracted markdown; not original HTML",
            "historical_availability_verified": False,
        }
        self.persist()
        if self.store:
            self.store.event(
                "web_page_saved",
                version_id=version_id,
                url=doc.url,
                **self.source_metadata[version_id],
            )
        return doc

    def read(self, version_id: str) -> DocumentVersion:
        if version_id not in self.documents:
            raise ValueError("page has not been extracted; use asynchronous aread")
        return self.documents[version_id]
