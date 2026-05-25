
import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest
from playwright.async_api import async_playwright


EXPECTED_TEXT = "Mocked clinical summary: atrial fibrillation is an arrhythmia."


def _build_mocked_app():
    mock_orch = MagicMock()
    mock_orch.answer.return_value = (
        "Cardiologist", EXPECTED_TEXT, "Source: mocked.txt",
    )
    def handle(q):
        spec, resp, ev = mock_orch.answer(q)
        return f"{spec}\n\n{resp}\n\n{ev}"
    with gr.Blocks() as demo:
        inp = gr.Textbox(label="query", elem_id="query_input")
        out = gr.Textbox(label="response", elem_id="response_output")
        btn = gr.Button("Submit", elem_id="submit_btn")
        btn.click(fn=handle, inputs=inp, outputs=out)
    return demo


@pytest.mark.asyncio
async def test_gradio_ui_returns_response_for_query():
    demo = _build_mocked_app()
    port = 7861
    thread = threading.Thread(
        target=lambda: demo.launch(server_port=port, prevent_thread_lock=False,
                                   quiet=True, show_error=True),
        daemon=True,
    )
    thread.start()
    time.sleep(2.0)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(f"http://127.0.0.1:{port}", timeout=15000)
                await page.fill("#query_input textarea", "What is AFib?")
                await page.click("#submit_btn")
                await page.wait_for_function(
                    f"document.querySelector('#response_output textarea')"
                    f".value.includes({EXPECTED_TEXT!r})",
                    timeout=10000,
                )
                response_text = await page.locator(
                    "#response_output textarea"
                ).input_value()
                assert EXPECTED_TEXT in response_text
            finally:
                await browser.close()
    finally:
        demo.close()
