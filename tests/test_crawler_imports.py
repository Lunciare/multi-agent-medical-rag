"""Confirm the scripts/data_processing/rag_crawler.py import surface is intact.

Replaces the previous print-and-swallow script (no assertions, bare-except)
with a real pytest case: every crawler dependency is imported via
`pytest.importorskip`, so the test cleanly skips on a base eval-only
environment (where PyMuPDF / Playwright are not installed) but **asserts the
crawler module API loudly when the deps are present**.

Why importorskip: the old script printed "fitz OK" / "Error: …" but never
failed CI. That hid a real defect — a broken crawler API. The fix here makes
the test actually assertive in the crawler environment, while not turning the
core eval suite red when crawler deps are absent.
"""

import importlib

import pytest


def test_crawler_imports():
    # Heavy I/O deps the crawler uses directly. `importorskip` skips this test
    # cleanly on environments that don't carry the crawler stack — and asserts
    # loudly on environments that do.
    pytest.importorskip("fitz",          reason="PyMuPDF not installed in this env")
    pytest.importorskip("requests",      reason="requests not installed in this env")
    pytest.importorskip("bs4",           reason="beautifulsoup4 not installed in this env")
    pytest.importorskip("playwright.async_api",
                        reason="playwright not installed in this env")

    # Crawler module itself plus its public TARGET_BOOKS constant.
    rag_crawler = importlib.import_module("scripts.data_processing.rag_crawler")
    assert hasattr(rag_crawler, "TARGET_BOOKS"), (
        "scripts.data_processing.rag_crawler.TARGET_BOOKS is missing; "
        "the crawler module API contract has been broken."
    )
