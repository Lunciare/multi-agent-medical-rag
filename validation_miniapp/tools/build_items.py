"""Build the served ``items.json`` from a team-filled raw-inputs file.

INTEGRITY (read tools/items_raw.schema.md). This tool is a deterministic transport +
formatting normalizer. It must NEVER fabricate, author, "improve", correct, reword, or
edit the substance of any case or answer, and must NEVER swap the RAG/vanilla mapping:

  - ``answer_rag_ru``     is derived ONLY from the raw RAG field   (case's RAG arm).
  - ``answer_vanilla_ru`` is derived ONLY from the raw vanilla field (case's vanilla arm).

The two arms are translated by the SAME ``translate`` callable and normalized by the SAME
rule set, in one code path, so nothing about style/formatting can leak which arm is which.
Every machine-produced output is DRAFT and must be human-reviewed before go-live; the build
emits a Markdown review bundle for exactly that.

CLI::

    python tools/build_items.py --raw tools/items_raw.json --mode manual \
        --out data/items.json --review data/items.review.md

    python tools/build_items.py --raw tools/items_raw.json --mode assisted \
        --translator my_pkg.translate:translate_ru --out data/items.json

Importable: ``build(raw_records, mode=..., translator=..., rules=...) -> BuildResult``.
"""

import argparse
import difflib
import importlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the EXISTING loader's validation rules (do not reinvent them).
_MINIAPP = Path(__file__).resolve().parent.parent
if str(_MINIAPP) not in sys.path:
    sys.path.insert(0, str(_MINIAPP))
from backend import items_loader  # noqa: E402


class BuildError(Exception):
    """Raised on any raw-input or assembly problem (names the bad item/field)."""


# --------------------------------------------------------------------------- #
# 1. Translation seam — ONE callable, applied identically to case + both arms. #
# --------------------------------------------------------------------------- #
def manual_passthrough(text):
    """Manual mode: Russian was supplied in the raw file; return it unchanged."""
    return text


def _import_callable(dotted):
    """Import 'module.path:callable' (or 'module.path.callable') for assisted mode."""
    if ":" in dotted:
        mod, name = dotted.split(":", 1)
    else:
        mod, name = dotted.rsplit(".", 1)
    try:
        obj = getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError) as e:
        raise BuildError(f"could not import translator {dotted!r}: {e}")
    if not callable(obj):
        raise BuildError(f"translator {dotted!r} is not callable")
    return obj


def make_translator(mode, translator_path=None):
    """Return (translate_fn, is_draft). Same fn is used for all three texts of an item."""
    if mode == "manual":
        return manual_passthrough, False
    if mode == "assisted":
        if not translator_path:
            raise BuildError(
                "assisted mode requires --translator 'module:callable' (a translator the "
                "team configures). The build never invents translations.")
        return _import_callable(translator_path), True
    raise BuildError(f"unknown mode {mode!r} (use 'manual' or 'assisted')")


# --------------------------------------------------------------------------- #
# 2. Normalization — conservative, content-preserving, identical to both arms. #
# --------------------------------------------------------------------------- #
# Each rule is (name, regex, replacement, sensitivity). sensitivity is one of:
#   "safe"             — structural strip that cannot change a medical claim
#                        (citation markers, templated headers, boilerplate footer).
#   "meaning_sensitive"— a minimal edit that touches words near a clinical claim
#                        (retrieval/"Context" framing). EVERY firing is flagged in the
#                        review bundle for explicit human sign-off.
# 3-tuples are accepted for back-compat and treated as "safe". Teams tune the list via
# --rules <json> (each entry [name, regex, replacement] or [name, regex, replacement, sens]).
SAFE = "safe"
MEANING_SENSITIVE = "meaning_sensitive"

DEFAULT_RULES = [
    # --- SAFE: inline citation markers --------------------------------------
    # [1], [1,2], [1-3], [ 12 ].
    ("inline_citation_brackets", r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]", "", SAFE),
    # Bare numeric refs in parens: (1), (2,3). Conservative (numbers only).
    ("inline_citation_parens_numeric", r"\((?:\s*\d+\s*(?:[-–,]\s*\d+\s*)*)\)", "", SAFE),
    # Source-file citations the RAG pipeline appends: "(Source: 0002.txt)",
    # "(Sources: 0001.txt, 0007.txt)". Remove the whole parenthetical (and a leading space).
    ("source_file_citation",
     r"(?i)\s*\((?:source|sources)\s*:\s*[^)]*\)", "", SAFE),

    # --- SAFE: templated section-header scaffolding -------------------------
    # Remove the fixed bold markdown headers the RAG role-prompt mandates
    # ("**Clinical Summary**", "**Evidence-Based Insights**", "**Limitations**"),
    # plus any trailing separator (— : -). KEEP the body text that follows.
    ("section_header_scaffolding",
     r"(?im)^[ \t]*\*\*\s*(?:Clinical\s+Summary|Evidence[ -]Based\s+Insights|Limitations)"
     r"\s*:?\s*\*\*[ \t]*[:–—-]?[ \t]*",
     "", SAFE),

    # --- SAFE: boilerplate disclaimer footer -------------------------------
    # The fixed RAG footer sentence. Whitespace-tolerant; strips optional quotes.
    ("boilerplate_disclaimer",
     r"(?i)[\"'«»]?\s*This\s+output\s+is\s+for\s+informational\s+use\s+by\s+medical\s+"
     r"professionals\s+only\s+and\s+does\s+not\s+constitute\s+a\s+diagnosis\s+or\s+"
     r"treatment\s+recommendation\.?\s*[\"'«»]?",
     "", SAFE),

    # --- MEANING_SENSITIVE: neutralize retrieval/"Context" meta-references --
    # These touch words next to clinical claims, so each firing is FLAGGED for human
    # sign-off. They drop the "Context"/"provided Context" framing while preserving the
    # clinical claim (and any declination, e.g. "treatment not provided"). Applied in
    # order: specific phrasings first, bare "the Context" last.
    # "According to the (provided) Context, X" -> "According to the available evidence, X"
    ("context_ref_according_to",
     r"(?i)according\s+to\s+the\s+(?:provided\s+)?context", "According to the available evidence",
     MEANING_SENSITIVE),
    # "... classification provided in the Context, ..." -> "... classification, ..."
    ("context_ref_provided_in",
     r"(?i)\s*(?:,\s*)?(?:as\s+)?provided\s+in\s+the\s+context", "", MEANING_SENSITIVE),
    # "... in the Context provided ..." -> "..."
    ("context_ref_in_the_context_provided",
     r"(?i)\s+in\s+the\s+context\s+provided", "", MEANING_SENSITIVE),
    # "The (provided) Context does not ..." -> "This does not ..." (declination kept).
    ("context_ref_the_context_does_not",
     r"\b[Tt]he\s+(?:provided\s+)?[Cc]ontext\s+does\s+not\b", "This does not", MEANING_SENSITIVE),
    # Remaining bare "the (provided) Context" (capitalised retrieval block; NOT "the context of").
    ("context_ref_the_context_bare",
     r"\b[Tt]he\s+(?:provided\s+)?Context\b(?!\s+of\b)", "the available information",
     MEANING_SENSITIVE),
]

# Objective no-coverage signal: the pipeline's standard refusal stub (specialist.py
# REFUSAL_RESPONSE). Items whose RAG answer IS this stub are excluded from the human
# study (a deterministic system-state signal, NOT a quality judgment). Matched on the
# raw ENGLISH answer, before translation. Configurable.
DEFAULT_NO_COVERAGE_MARKERS = (
    "Insufficient evidence in the current knowledge base to address this specific query.",
)


def _unpack_rule(r):
    if len(r) == 4:
        return r[0], r[1], r[2], r[3]
    if len(r) == 3:
        return r[0], r[1], r[2], SAFE
    raise BuildError(f"bad rule (need [name, regex, repl] or [name, regex, repl, sensitivity]): {r!r}")


def normalize(text, rules):
    """Apply strip ``rules`` then tidy only the whitespace they disturbed.

    Returns ``(out, edits)`` where ``edits`` lists every rule that fired:
    ``[{"rule", "sensitivity", "count"}]``. Content-preserving: it removes matched tells
    (and, for meaning_sensitive rules, makes minimal framing edits) and collapses leftover
    whitespace. It does NOT truncate, summarize, reword medical claims, or pad.
    """
    out = text
    edits = []
    for r in rules:
        name, pattern, repl, sens = _unpack_rule(r)
        out, n = re.subn(pattern, repl, out)
        if n:
            edits.append({"rule": name, "sensitivity": sens, "count": n})
    # Whitespace tidy limited to artifacts of removal (no semantic change):
    out = re.sub(r"[ \t]{2,}", " ", out)            # collapse runs of spaces/tabs
    out = re.sub(r"[ \t]+([,.;:!?»)])", r"\1", out)  # drop space before punctuation
    out = re.sub(r"([(«])[ \t]+", r"\1", out)        # drop space after opening bracket/quote
    out = "\n".join(line.rstrip() for line in out.split("\n"))  # trailing ws per line
    out = re.sub(r"\n{3,}", "\n\n", out)            # collapse 3+ blank lines
    return out.strip(), edits


def _diff(before, after):
    """Unified line diff of one answer before vs after normalization (for humans)."""
    if before == after:
        return "(no change — normalization removed nothing from this answer)"
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile="before_normalization", tofile="after_normalization", lineterm="")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Assembly + validation                                                     #
# --------------------------------------------------------------------------- #
# Per-mode raw source fields. The SAME positional meaning (case / rag / vanilla)
# regardless of mode, so the arm mapping cannot drift.
_SOURCE_FIELDS = {
    "manual":   {"case": "case_ru", "rag": "answer_rag_ru", "vanilla": "answer_vanilla_ru"},
    "assisted": {"case": "case_en", "rag": "answer_rag_en", "vanilla": "answer_vanilla_en"},
}
_COMMON_FIELDS = ("item_id", "gold_specialty", "available_specialties", "routed_specialty")


@dataclass
class AnswerTrace:
    # ORDER OF OPERATIONS: source_text --normalize--> normalized_src --translate--> served.
    arm: str                # "rag" | "vanilla"
    raw_field: str          # exact raw field this came from (provenance)
    source_text: str        # raw source (EN in assisted, RU in manual)
    normalized_src: str     # after normalize() on the SOURCE (BEFORE translation)
    served: str             # final served text (RU): translate(normalized_src)
    norm_diff: str          # diff(source_text, normalized_src) — shows tells removed
    edits: list = field(default_factory=list)   # every rule that fired (name/sensitivity/count)

    @property
    def meaning_sensitive_edits(self):
        return [e for e in self.edits if e.get("sensitivity") == MEANING_SENSITIVE]

    # Back-compat aliases (older callers/tests).
    @property
    def diff(self):
        return self.norm_diff

    @property
    def normalized_ru(self):
        return self.served


@dataclass
class ItemTrace:
    item_id: str
    mode: str
    is_draft: bool
    case_source: str
    case_ru: str
    rag: AnswerTrace
    vanilla: AnswerTrace


@dataclass
class BuildResult:
    items: list                      # the served items.json content (validated)
    traces: list = field(default_factory=list)  # per-item provenance + diffs
    review_md: str = ""
    mode: str = ""
    is_draft: bool = False
    rules: list = field(default_factory=list)
    excluded_no_coverage: list = field(default_factory=list)  # [{item_id, matched_marker, ...}]


def _require(record, fieldname, label):
    val = record.get(fieldname)
    if isinstance(val, str):
        if val.strip() == "":
            raise BuildError(f"{label}: field '{fieldname}' is empty")
    elif val is None:
        raise BuildError(f"{label}: field '{fieldname}' is missing")
    return val


def _match_no_coverage(answer_rag_text, markers):
    """Return the matched marker if ``answer_rag_text`` IS the pipeline's no-coverage stub.

    Deterministic system-state signal (not a quality judgment). Matched on the raw English,
    whitespace-normalized, as a substring — the real stub is the entire answer, so this only
    fires on stub-only answers; partial-but-substantive answers (which merely *mention* a gap)
    are NOT excluded.
    """
    hay = " ".join(str(answer_rag_text).split())
    for m in markers:
        needle = " ".join(str(m).split())
        if needle and needle in hay:
            return m
    return None


def build(raw_records, *, mode, translator=None, rules=None,
          exclude_no_coverage=False, no_coverage_markers=None):
    """Convert raw records -> validated items + review bundle. Pure/importable.

    ORDER: normalize(source) -> translate(normalized) -> assemble (so Russian comes out
    clean). ``translator`` may be a callable or a dotted path string. ``rules`` defaults to
    DEFAULT_RULES. With ``exclude_no_coverage=True``, items whose RAG answer matches a
    ``no_coverage_markers`` stub (raw English, pre-translation) are EXCLUDED from the study
    and recorded (never silently dropped). Raises BuildError naming the bad item/field.
    """
    if rules is None:
        rules = DEFAULT_RULES
    if no_coverage_markers is None:
        no_coverage_markers = list(DEFAULT_NO_COVERAGE_MARKERS)
    if callable(translator):
        translate_fn, is_draft = translator, (mode == "assisted")
    else:
        translate_fn, is_draft = make_translator(mode, translator)
    src = _SOURCE_FIELDS.get(mode)
    if src is None:
        raise BuildError(f"unknown mode {mode!r} (use 'manual' or 'assisted')")

    if not isinstance(raw_records, list) or not raw_records:
        raise BuildError("raw inputs must be a non-empty JSON array")

    items = []
    traces = []
    excluded = []
    seen = set()
    for i, rec in enumerate(raw_records):
        label = f"raw[{i}]"
        if not isinstance(rec, dict):
            raise BuildError(f"{label}: must be a JSON object")
        iid = rec.get("item_id")
        if isinstance(iid, str) and iid.strip():
            label = f"item_id={iid!r}"
        for f in _COMMON_FIELDS:
            _require(rec, f, label)
        if iid in seen:
            raise BuildError(f"{label}: duplicate item_id")
        seen.add(iid)
        for f in src.values():
            _require(rec, f, label)

        case_src = rec[src["case"]]
        rag_src = rec[src["rag"]]
        van_src = rec[src["vanilla"]]

        # --- working-case filter: exclude no-coverage stubs (raw English, pre-translate) ---
        if exclude_no_coverage:
            matched = _match_no_coverage(rag_src, no_coverage_markers)
            if matched is not None:
                excluded.append({
                    "item_id": rec["item_id"],
                    "matched_marker": matched,
                    "gold_specialty": rec["gold_specialty"],
                    "routed_specialty": rec["routed_specialty"],
                    "answer_rag_en": rag_src,
                })
                continue  # recorded, not silently dropped; skip translation/assembly

        # --- normalize FIRST (on the SOURCE), then translate the normalized text ---
        # SAME rules object applied identically to both answers (vanilla -> usually no-op).
        rag_norm, rag_edits = normalize(rag_src, rules)
        van_norm, van_edits = normalize(van_src, rules)
        rag_ru = translate_fn(rag_norm)
        van_ru = translate_fn(van_norm)
        case_ru = translate_fn(case_src)  # case shown once (neutral); not normalized.

        item = {
            "item_id": rec["item_id"],
            "case_ru": case_ru,
            "gold_specialty": rec["gold_specialty"],
            "routed_specialty": rec["routed_specialty"],
            "available_specialties": rec["available_specialties"],
            "answer_rag_ru": rag_ru,
            "answer_vanilla_ru": van_ru,
        }
        items.append(item)
        traces.append(ItemTrace(
            item_id=rec["item_id"], mode=mode, is_draft=is_draft,
            case_source=case_src, case_ru=case_ru,
            rag=AnswerTrace("rag", src["rag"], rag_src, rag_norm, rag_ru,
                            _diff(rag_src, rag_norm), rag_edits),
            vanilla=AnswerTrace("vanilla", src["vanilla"], van_src, van_norm, van_ru,
                                _diff(van_src, van_norm), van_edits),
        ))

    if not items:
        raise BuildError("no items remain after the no-coverage filter (nothing to build)")

    # Final gate: the EXISTING app loader must accept this file.
    try:
        items_loader.validate_items(items)
    except items_loader.ItemsValidationError as e:
        raise BuildError(f"assembled items.json failed loader validation: {e}")

    review_md = render_review(traces, mode=mode, is_draft=is_draft, rules=rules,
                              excluded=excluded)
    return BuildResult(items=items, traces=traces, review_md=review_md,
                       mode=mode, is_draft=is_draft, rules=list(rules),
                       excluded_no_coverage=excluded)


# --------------------------------------------------------------------------- #
# 4. Human-review bundle (single Markdown file)                                #
# --------------------------------------------------------------------------- #
def render_review(traces, *, mode, is_draft, rules, excluded=None):
    excluded = excluded or []
    L = []
    L.append("# items.json — human review bundle")
    L.append("")
    banner = ("DRAFT — MACHINE-TRANSLATED" if is_draft
              else "MANUAL RU — NO MACHINE TRANSLATION (still verify mapping + normalization)")
    L.append(f"> **{banner}.** Do NOT go live until a Russian-speaking reviewer signs off "
             f"**every** item below. This tool only transports + normalizes formatting; it "
             f"does not author or correct content. Order of operations per answer: "
             f"**normalize (English) → translate → serve**.")
    L.append("")
    L.append(f"- mode: `{mode}`")
    L.append(f"- normalization rules applied (identically to both arms): "
             f"{[r[0] for r in rules]}")
    L.append(f"- items kept: {len(traces)}")
    if excluded:
        L.append(f"- items EXCLUDED (no-coverage stub; objective system-state signal, not a "
                 f"quality judgment): {len(excluded)}")
        for ex in excluded:
            L.append(f"  - `{ex['item_id']}` (gold={ex['gold_specialty']}, "
                     f"routed={ex['routed_specialty']}) — matched: \"{ex['matched_marker']}\"")
    L.append("")
    for t in traces:
        L.append("---")
        L.append(f"## {t.item_id}  —  DRAFT, NEEDS REVIEW")
        L.append("")
        L.append("### Case")
        if mode == "assisted":
            L.append("**Source (EN):**"); L.append(""); L.append(_q(t.case_source)); L.append("")
            L.append("**Translated (RU, draft):**")
        else:
            L.append("**RU (as supplied):**")
        L.append(""); L.append(_q(t.case_ru)); L.append("")
        for ans in (t.rag, t.vanilla):
            arm = "RAG" if ans.arm == "rag" else "VANILLA"
            L.append(f"### Answer — arm: {arm}  (from raw field `{ans.raw_field}`)")
            if mode == "assisted":
                L.append("**Source (EN, raw):**"); L.append(""); L.append(_q(ans.source_text)); L.append("")
                L.append("**Normalized (EN, after tell-strip — what gets translated):**")
                L.append(""); L.append(_q(ans.normalized_src)); L.append("")
                L.append("**Served (RU, translated draft):**")
            else:
                L.append("**Served (RU, normalized):**")
            L.append(""); L.append(_q(ans.served)); L.append("")
            L.append("**Normalization diff on the source (must show tells removed ONLY):**")
            L.append(""); L.append("```diff"); L.append(ans.norm_diff); L.append("```"); L.append("")
            if ans.meaning_sensitive_edits:
                fired = ", ".join(f"`{e['rule']}` ×{e['count']}" for e in ans.meaning_sensitive_edits)
                L.append(f"> ⚠️ **MEANING-SENSITIVE EDITS — REQUIRE EXPLICIT SIGN-OFF.** "
                         f"Retrieval/\"Context\" framing was edited ({fired}). Confirm in the diff "
                         f"above that ONLY the retrieval framing changed — the clinical claim, any "
                         f"numbers/drugs, and any declination (e.g. \"treatment not provided\") "
                         f"must be unchanged.")
                L.append("")
        item_has_sensitive = bool(t.rag.meaning_sensitive_edits or t.vanilla.meaning_sensitive_edits)
        L.append("### Reviewer checklist")
        L.append("- [ ] Translation faithful (case + both answers; no meaning added/lost)?")
        L.append("- [ ] Normalization removed **tells only** (no medical content changed)?")
        if item_has_sensitive:
            L.append("- [ ] ⚠️ Each meaning-sensitive \"Context\" edit changed ONLY the retrieval "
                     "framing — the clinical claim and any declination survived?")
        L.append("- [ ] RAG/vanilla mapping correct "
                 "(RAG answer = team's RAG output; vanilla answer = team's vanilla output)?")
        L.append("")
    return "\n".join(L)


def _q(text):
    """Quote a block for Markdown (prefix each line with '> ')."""
    return "\n".join("> " + line if line else ">" for line in str(text).split("\n"))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def load_raw(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BuildError(f"raw inputs file not found: {path}")
    except json.JSONDecodeError as e:
        raise BuildError(f"raw inputs file is not valid JSON: {e}")
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description="Build validated items.json from raw team inputs.")
    p.add_argument("--raw", required=True, help="raw inputs JSON (see tools/items_raw.schema.md)")
    p.add_argument("--mode", choices=("manual", "assisted"), required=True)
    p.add_argument("--translator", default=None,
                   help="assisted mode: 'module:callable' translator the team provides")
    p.add_argument("--rules", default=None,
                   help="optional JSON file: list of [name, pattern, replacement] strip rules")
    p.add_argument("--out", default=str(_MINIAPP / "data" / "items.json"))
    p.add_argument("--review", default=None,
                   help="review bundle path (default: alongside --out as *.review.md)")
    p.add_argument("--exclude-no-coverage", action="store_true",
                   help="exclude items whose RAG answer is the no-coverage stub (human study)")
    p.add_argument("--no-coverage-marker", action="append", default=None,
                   help="override/add no-coverage stub marker(s); repeatable")
    args = p.parse_args(argv)

    rules = DEFAULT_RULES
    if args.rules:
        loaded = json.loads(Path(args.rules).read_text(encoding="utf-8"))
        rules = [tuple(r) for r in loaded]
    markers = args.no_coverage_marker or list(DEFAULT_NO_COVERAGE_MARKERS)

    try:
        raw = load_raw(args.raw)
        result = build(raw, mode=args.mode, translator=args.translator, rules=rules,
                       exclude_no_coverage=args.exclude_no_coverage,
                       no_coverage_markers=markers)
    except BuildError as e:
        print(f"BUILD FAILED: {e}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.items, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    review_path = Path(args.review) if args.review else out_path.with_suffix(".review.md")
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(result.review_md, encoding="utf-8")

    flag = "DRAFT (machine-translated)" if result.is_draft else "manual RU"
    # Exclusions: log every excluded item_id + matched marker (never silently dropped).
    if args.exclude_no_coverage:
        print(f"Excluded {len(result.excluded_no_coverage)} no-coverage item(s):")
        for ex in result.excluded_no_coverage:
            print(f"  - {ex['item_id']} (gold={ex['gold_specialty']}, "
                  f"routed={ex['routed_specialty']}) matched: \"{ex['matched_marker']}\"")
    per_spec = {}
    for it in result.items:
        per_spec[it["gold_specialty"]] = per_spec.get(it["gold_specialty"], 0) + 1
    print(f"Wrote {out_path} ({len(result.items)} items, {flag}) — passed loader validation.")
    print(f"Final per-specialty (gold) counts: {per_spec}")
    print(f"Wrote review bundle {review_path}")
    print("NEXT: a Russian-speaking reviewer must sign off every item before go-live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
