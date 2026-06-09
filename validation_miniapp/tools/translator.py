"""English -> Russian translator adapter for ``build_items.py --mode assisted``.

Pass it as ``--translator tools.translator:translate``. ``build_items`` applies this callable
**identically** to the case and BOTH answers, so it must treat every text the same way — it
has no per-arm or per-content branching (blinding-critical).

Design:
  - Reuses the project's EXISTING Yandex credentials/client (``multi-agent_system/settings.py``;
    same ``.env``). No new key is introduced.
  - Minimal, translate-ONLY system prompt; ``temperature=0`` for stable re-runs.
  - FAITHFUL: the adapter never summarizes, omits, adds, reorders, or editorializes. It returns
    the backend's output **verbatim** (pure pass-through) — formatting/tell removal and human
    review happen in the separate normalization + review steps, not here.
  - Empty string -> empty string (no backend call). Any backend failure RAISES
    ``TranslatorError`` so ``build_items`` surfaces it instead of recording a garbled result.

TRADEOFF (flagged): an LLM translator may smooth style (which incidentally helps blinding) but
can drift on content. Faithfulness is the priority; the human review bundle is the safety net.
"""

import sys
from pathlib import Path

# Translate-ONLY instruction. Faithful + complete: translate everything into Russian,
# leaving NO English fragments and never mixing scripts inside a word.
SYSTEM_PROMPT = (
    "You are a professional medical translator. Translate the user's message from English "
    "into Russian.\n"
    "Output ONLY the translation — no preamble, notes, explanations, or surrounding quotation "
    "marks.\n"
    "Faithfulness rules (do NOT break):\n"
    "- Translate EVERY English word and phrase into Russian. Leave NO English words or "
    "fragments in the output (e.g. not 'preoperative', not 'postoperative').\n"
    "- Never mix Cyrillic and Latin letters within a single word.\n"
    "- Transliterate proper names correctly into Russian (e.g. Balthazar -> Бальтазар).\n"
    "- For medical abbreviations/acronyms, render them consistently: translate to the standard "
    "Russian term where one exists, otherwise keep the Latin acronym consistently throughout; "
    "do not switch back and forth.\n"
    "- Keep all numbers, units, dosages, lab values, and drug names accurate and consistent.\n"
    "- Do NOT summarize, omit, add, reorder, soften, or explain anything. Preserve the full "
    "content, structure, line breaks, and any hedging or declination exactly as written."
)


class TranslatorError(Exception):
    """Raised on any backend/transport failure or an unusable backend response."""


def build_messages(text):
    """Chat messages for a faithful EN->RU translation. ``text`` is passed through verbatim."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def _load_settings():
    """Lazily import the project's settings (Yandex client + model), loading its .env so the
    creds resolve regardless of the current working directory. Lazy so this module imports
    offline (the smoke test injects a mock backend and never reaches here)."""
    repo_root = Path(__file__).resolve().parents[2]
    mas = repo_root / "multi-agent_system"
    try:
        from dotenv import load_dotenv
        env_file = mas / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    except Exception:
        pass  # settings.py also calls load_dotenv(); env vars may already be present.
    if str(mas) not in sys.path:
        sys.path.insert(0, str(mas))
    import settings  # noqa: E402
    return settings


def _default_complete(messages):
    """Real backend: the project's Yandex client (same creds as multi-agent_system)."""
    settings = _load_settings()
    resp = settings.client.chat.completions.create(
        model=settings.AGENT_MODEL,
        messages=messages,
        temperature=0,
        extra_headers={"x-folder-id": settings.YANDEX_PROJECT_ID},
    )
    content = resp.choices[0].message.content
    if content is None:
        raise TranslatorError("translator backend returned no content")
    return content


def translate(text, *, complete=None):
    """Translate ``text`` EN->Russian and return it verbatim.

    ``complete`` is an injectable backend ``callable(messages) -> str`` (the smoke test passes a
    mock); production uses the Yandex client. Empty string returns empty (no call). Any backend
    failure raises ``TranslatorError`` — never a partial/garbled result.
    """
    if not isinstance(text, str):
        raise TypeError(f"translate() expects str, got {type(text).__name__}")
    if text == "":
        return ""

    backend = complete if complete is not None else _default_complete
    try:
        out = backend(build_messages(text))
    except TranslatorError:
        raise
    except Exception as e:
        raise TranslatorError(f"translation backend failed: {type(e).__name__}: {e}") from e

    if not isinstance(out, str):
        raise TranslatorError(f"translator backend returned non-string: {type(out).__name__}")
    # Pure pass-through: the adapter itself does not edit/strip/mangle the translation.
    return out
