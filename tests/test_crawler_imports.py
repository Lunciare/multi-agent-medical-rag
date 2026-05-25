
import importlib

import pytest


def test_crawler_imports():
    pytest.importorskip("fitz",          reason="PyMuPDF not installed in this env")
    pytest.importorskip("requests",      reason="requests not installed in this env")
    pytest.importorskip("bs4",           reason="beautifulsoup4 not installed in this env")
    pytest.importorskip("playwright.async_api",
                        reason="playwright not installed in this env")

    rag_crawler = importlib.import_module("scripts.data_processing.rag_crawler")
    assert hasattr(rag_crawler, "TARGET_BOOKS"), (
        "scripts.data_processing.rag_crawler.TARGET_BOOKS is missing; "
        "the crawler module API contract has been broken."
    )
