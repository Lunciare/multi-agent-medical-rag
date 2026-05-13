import pytest
from playwright.async_api import async_playwright


@pytest.mark.asyncio
async def test_chromium_renders_blank_page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto("about:blank")
            assert page.url == "about:blank"
            title = await page.title()
            assert isinstance(title, str)
        finally:
            await browser.close()
