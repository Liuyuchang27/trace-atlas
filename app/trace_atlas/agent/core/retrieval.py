"""Run-scoped search and reading adapters for Trace Atlas."""

from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import Future
import logging
import os
from threading import Event, Lock
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from ..concurrency import check_cancelled

active_retrieval = ContextVar("trace_atlas_retrieval", default=None)


@contextmanager
def retrieval_scope(backend):
    token = active_retrieval.set(backend)
    try:
        yield
    finally:
        active_retrieval.reset(token)


def _post(url, key, payload):
    check_cancelled()
    response = requests.post(url, headers={"Authorization": "Bearer " + key}, json=payload, timeout=45)
    response.raise_for_status()
    return response.json()


class WebRetrieval:
    def __init__(self):
        self.namespace = uuid4().hex
        self._reads = {}
        self._lock = Lock()

    def read(self, url):
        with self._lock:
            existing = self._reads.get(url)
            if existing is None:
                self._reads[url] = pending = Future()
        if existing is not None:
            return existing.result()
        try:
            content = self._fetch(url)
            pending.set_result(content or "[Error: no extractable page text]")
            if not content:
                with self._lock:
                    self._reads.pop(url, None)
            return pending.result()
        except BaseException as exc:
            pending.set_exception(exc)
            with self._lock:
                self._reads.pop(url, None)
            raise


class DeepSeekRetrieval(WebRetrieval):
    name = "deepseek"
    search_query_limit = 128

    def __init__(self, fallback=None):
        super().__init__()
        self.key = os.environ.get("DEEPSEEK_API_KEY")
        self._indexed_content = {}
        self.fallback = fallback
        self.quota_exhausted = Event()
        if not self.key:
            raise ValueError("Set DEEPSEEK_API_KEY before using DeepSeek Search and Fetch")

    def _can_fallback(self, exc):
        if not self.fallback:
            return False
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        status = exc.response.status_code if exc.response is not None else 0
        if status == 402:
            self.quota_exhausted.set()
            logging.warning("DeepSeek search quota exhausted. Using Tavily for the rest of this run.")
        return status in {402, 429} or 500 <= status < 600

    def search(self, query, limit):
        if self.quota_exhausted.is_set():
            return self.fallback.search(query, limit)
        try:
            data = _post("https://api.deepseek.com/v1/tools/search", self.key, {
                "query": query, "count": limit, "need_content": True,
            })
        except requests.RequestException as exc:
            if self._can_fallback(exc):
                return self.fallback.search(query, limit)
            raise
        results = data.get("results", [])
        with self._lock:
            self._indexed_content.update((row["url"], row["content"]) for row in results if row.get("content"))
        return [
            {
                "index": i, "url": row["url"], "title": row.get("title") or row["url"],
                "snippet": row.get("summary") or "", "domain": urlsplit(row["url"]).netloc,
                "date_published": row.get("publish_time"), "retriever": self.name,
            }
            for i, row in enumerate(results)
        ]

    def _fetch(self, url):
        with self._lock:
            indexed = self._indexed_content.get(url)
        if self.quota_exhausted.is_set():
            return indexed or self.fallback.read(url)
        try:
            live = _post("https://api.deepseek.com/v1/tools/fetch", self.key, {"url": url}).get("content")
        except requests.RequestException as exc:
            can_fallback = self._can_fallback(exc)
            if indexed:
                return indexed
            if can_fallback:
                return self.fallback.read(url)
            raise
        return live or indexed or ""

    def images(self, query):
        return self.fallback.images(query) if self.fallback else []


class TavilyRetrieval(WebRetrieval):
    name = "tavily"
    search_query_limit = None

    def __init__(self):
        super().__init__()
        self.key = os.environ.get("TAVILY_API_KEY")
        if not self.key:
            raise ValueError("Set TAVILY_API_KEY before using Tavily")

    def _tavily(self, operation, payload):
        return _post("https://api.tavily.com/" + operation, self.key, payload)

    def search(self, query, limit):
        data = self._tavily("search", {"query": query, "max_results": limit, "include_published_date": True})
        return [
            {
                "index": i, "url": row["url"], "title": row.get("title") or row["url"],
                "snippet": row.get("content") or "", "domain": urlsplit(row["url"]).netloc,
                "date_published": row.get("published_date"), "retriever": self.name,
            }
            for i, row in enumerate(data.get("results", []))
        ]

    def images(self, query):
        data = self._tavily("search", {
            "query": query, "max_results": 10, "include_images": True, "include_image_descriptions": True,
        })
        return [
            {"url": image["url"], "description": image.get("description") or ""}
            for image in data.get("images", [])
            if isinstance(image, dict) and urlsplit(image.get("url") or "").scheme in {"https", "http"}
        ]

    def _fetch(self, url):
        data = self._tavily("extract", {"urls": [url], "extract_depth": "advanced"})
        return next((row["raw_content"] for row in data.get("results", []) if row.get("raw_content")), "")
