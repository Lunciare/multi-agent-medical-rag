import sys
from pathlib import Path

_MINIAPP = Path(__file__).resolve().parent.parent
if str(_MINIAPP) not in sys.path:
    sys.path.insert(0, str(_MINIAPP))

from tools import translator  # noqa: E402

_checks = []


def check(label, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    _checks.append((label, bool(cond)))
    return bool(cond)


def section(t):
    print("\n" + "=" * 70); print(t); print("=" * 70)


def run():
    section("STEP 1: translate() returns the mocked RU for a given EN")
    # Fixed mock: a known EN -> known RU mapping.
    def mock_fixed(messages):
        return "Сердечная недостаточность с фракцией выброса 30%."
    en = "Heart failure with an ejection fraction of 30%."
    out = translator.translate(en, complete=mock_fixed)
    print(f"  EN: {en}")
    print(f"  RU: {out}")
    check("returns exactly the backend's RU output",
          out == "Сердечная недостаточность с фракцией выброса 30%.")

    section("STEP 2: pure pass-through (exact text forwarded; output verbatim)")
    captured = {}

    def mock_echo(messages):
        # Record what the adapter forwarded, and echo the user content back.
        captured["messages"] = messages
        user = messages[-1]["content"]
        return f"RU::{user}"

    for sample in ("Atrial fibrillation, HbA1c 7.8%, metformin 500 mg.",
                   "**Clinical Summary** [1] the Context does not provide treatment."):
        r = translator.translate(sample, complete=mock_echo)
        print(f"  in : {sample}")
        print(f"  out: {r}")
        check(f"adapter forwards EXACT text to backend ({sample[:24]}...)",
              captured["messages"][-1]["content"] == sample)
        check("adapter returns backend output verbatim (no mangling)", r == f"RU::{sample}")

    # System prompt is translate-only and the user message is the raw text (no wrapping).
    msgs = translator.build_messages("hello")
    check("system prompt is translate-only", "Translate" in msgs[0]["content"] and "ONLY" in msgs[0]["content"])
    check("system prompt forbids summarizing/adding",
          "summarize" in msgs[0]["content"].lower() and "omit" in msgs[0]["content"].lower())
    check("user message is the raw text unchanged", msgs[-1]["content"] == "hello")

    section("STEP 3: empty string handled WITHOUT calling the backend")
    def mock_must_not_run(messages):
        raise AssertionError("backend should not be called for empty string")
    empty_out = translator.translate("", complete=mock_must_not_run)
    print(f"  translate('') -> {empty_out!r}")
    check("empty string returns empty", empty_out == "")

    section("STEP 4: backend failure RAISES (never a silent/partial result)")
    def mock_boom(messages):
        raise RuntimeError("simulated Yandex API outage")
    raised = False
    try:
        translator.translate("anything", complete=mock_boom)
    except translator.TranslatorError as e:
        raised = True
        print(f"  raised TranslatorError: {e}")
    check("backend failure raises TranslatorError", raised)

    # A non-string backend response is also rejected (not returned as garbage).
    raised_nonstr = False
    try:
        translator.translate("x", complete=lambda m: None)
    except translator.TranslatorError as e:
        raised_nonstr = True
        print(f"  raised on non-string response: {e}")
    check("non-string backend response raises TranslatorError", raised_nonstr)

    section("STEP 5: identical treatment regardless of content (blinding)")
    # The same adapter path handles 'RAG-looking' and 'vanilla-looking' text identically.
    rag_like = translator.translate("RAG arm answer", complete=mock_echo)
    van_like = translator.translate("vanilla arm answer", complete=mock_echo)
    print(f"  rag_like -> {rag_like}")
    print(f"  van_like -> {van_like}")
    check("no per-content branching (both wrapped identically)",
          rag_like == "RU::RAG arm answer" and van_like == "RU::vanilla arm answer")

    section("RESULT")
    failed = [lbl for lbl, ok in _checks if not ok]
    print(f"{len(_checks) - len(failed)}/{len(_checks)} checks passed")
    if failed:
        print("FAILED CHECKS:")
        for lbl in failed:
            print(f"  - {lbl}")
    print("=" * 70)
    return not failed


def test_smoke_translator():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
