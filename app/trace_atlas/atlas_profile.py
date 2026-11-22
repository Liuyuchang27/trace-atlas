"""Shared Trace Atlas launch configuration for web and benchmark runs."""

import os


def profile(*, web_fallback=False):
    pro = os.getenv("ATLAS_PRO_MODEL", "deepseek-v4-pro")
    flash = os.getenv("ATLAS_FLASH_MODEL", "deepseek-flash")
    return {
        "llm_provider": "deepseek",
        "llm_model": pro,
        "searcher_model": pro,
        "cognitive_model": flash,
        "reader_model": flash,
        "prefilter_model": flash,
        "quality_model": None,
        "writer_model": pro,
        "search_engine": os.getenv("ATLAS_SEARCH_ENGINE", "auto" if web_fallback else "deepseek"),
        "max_results": int(os.getenv("ATLAS_SEARCH_RESULTS", "12")),
        "include_search_refs": False,
        "atlas_mode": True,
    }
