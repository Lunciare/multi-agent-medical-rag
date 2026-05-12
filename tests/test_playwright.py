import asyncio
from playwright.async_api import async_playwright

async def main():
    print("Starting Playwright...")
    try:
        async with async_playwright() as p:
            print("Launching chromium...")
            browser = await p.chromium.launch(headless=True)
            print("Browser launched.")
            await browser.close()
            print("Browser closed.")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
