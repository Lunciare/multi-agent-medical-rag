import json
import sys
import tempfile
from pathlib import Path

_MINIAPP = Path(__file__).resolve().parent.parent
if str(_MINIAPP) not in sys.path:
    sys.path.insert(0, str(_MINIAPP))

from backend import items_loader  # noqa: E402
from tools import build_items  # noqa: E402

FIXTURE = _MINIAPP / "tests" / "fixtures" / "items_raw.dummy.json"

_checks = []


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    _checks.append((label, bool(condition)))
    return bool(condition)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

    section("STEP 1: build (manual mode) via the importable API")
    result = build_items.build(raw, mode="manual")
    print(f"  mode={result.mode} is_draft={result.is_draft} "
          f"rules={[r[0] for r in result.rules]} items={len(result.items)}")
    check("manual mode is not flagged machine-translated", result.is_draft is False)
    check("built one item per raw record", len(result.items) == len(raw))

    section("STEP 2: produced items.json (verbatim) + EXISTING-loader validation")
    tmp = Path(tempfile.mkdtemp())
    out_path = tmp / "items.json"
    review_path = tmp / "items.review.md"
    rc = build_items.main([
        "--raw", str(FIXTURE), "--mode", "manual",
        "--out", str(out_path), "--review", str(review_path),
    ])
    check("CLI main() exited 0", rc == 0)
    print("----- items.json BEGIN -----")
    print(out_path.read_text(encoding="utf-8"), end="")
    print("----- items.json END -----")
    loaded, by_id = items_loader.load_and_validate(out_path)
    check("items.json passes items_loader.load_and_validate", len(loaded) == len(raw))

    section("STEP 3: RAG/vanilla mapping traces to the correct input fields (no swap)")
    raw_by_id = {r["item_id"]: r for r in raw}
    for tr in result.traces:
        item = by_id[tr.item_id]
        rec = raw_by_id[tr.item_id]
        # Provenance: each arm declares exactly which raw field it came from.
        prov_ok = (tr.rag.raw_field == "answer_rag_ru"
                   and tr.vanilla.raw_field == "answer_vanilla_ru")
        # Content: distinctive markers must land in the matching output field only.
        rag_ok = ("RAG_ARM_MARKER" in item["answer_rag_ru"]
                  and "VANILLA_ARM_MARKER" not in item["answer_rag_ru"])
        van_ok = ("VANILLA_ARM_MARKER" in item["answer_vanilla_ru"]
                  and "RAG_ARM_MARKER" not in item["answer_vanilla_ru"])
        # And the served text derives from the raw arm field (marker present in raw).
        src_ok = ("RAG_ARM_MARKER" in rec["answer_rag_ru"]
                  and "VANILLA_ARM_MARKER" in rec["answer_vanilla_ru"])
        print(f"  {tr.item_id}: rag<-{tr.rag.raw_field}  vanilla<-{tr.vanilla.raw_field}  "
              f"| rag_marker_ok={rag_ok} vanilla_marker_ok={van_ok}")
        check(f"{tr.item_id}: arm provenance fields correct", prov_ok)
        check(f"{tr.item_id}: RAG output traces to RAG input only", rag_ok and src_ok)
        check(f"{tr.item_id}: vanilla output traces to vanilla input only", van_ok)

    section("STEP 4: identical normalization applied to BOTH arms")
    print(f"  rules used (same object for both arms): {[r[0] for r in result.rules]}")
    for tr in result.traces:
        item = by_id[tr.item_id]
        rec = raw_by_id[tr.item_id]
        # The '[1]' citation tell is present in BOTH raw answers...
        tell_in_raw = ("[1]" in rec["answer_rag_ru"]) and ("[1]" in rec["answer_vanilla_ru"])
        # ...and removed from BOTH served answers (identical rule application).
        tell_gone = ("[1]" not in item["answer_rag_ru"]) and ("[1]" not in item["answer_vanilla_ru"])
        # The case is shown once (neutral) and is NOT normalized: its tell remains,
        # proving normalization targeted the answers, not a global pass.
        case_untouched = "[1]" in item["case_ru"] if "[1]" in rec["case_ru"] else True
        print(f"  {tr.item_id}: tell_in_both_raw={tell_in_raw} tell_removed_from_both={tell_gone} "
              f"case_tell_preserved={case_untouched}")
        check(f"{tr.item_id}: citation tell removed from BOTH arms identically",
              tell_in_raw and tell_gone)
        check(f"{tr.item_id}: case left un-normalized (answers normalized, not global)",
              case_untouched)

    # Show one before/after diff verbatim so the format is visible.
    print("\n  --- example normalization diff (raw-001 RAG arm) ---")
    print(result.traces[0].rag.diff)

    section("STEP 5: human-review bundle generated")
    review_text = review_path.read_text(encoding="utf-8")
    check("review bundle file exists", review_path.is_file())
    check("review bundle carries a DRAFT / sign-off banner",
          "DRAFT" in review_text and "signs off" in review_text.lower())
    check("review bundle has the per-item mapping checklist line",
          review_text.count("RAG/vanilla mapping correct") == len(raw))
    check("review bundle labels arm provenance (raw field) for both arms",
          "from raw field `answer_rag_ru`" in review_text
          and "from raw field `answer_vanilla_ru`" in review_text)
    print("----- review bundle (first 40 lines) -----")
    print("\n".join(review_text.split("\n")[:40]))
    print("----- (review bundle truncated) -----")

    run_tell_normalization()
    run_filter_and_ordering()

    section("RESULT")
    failed = [lbl for lbl, ok in _checks if not ok]
    print(f"{len(_checks) - len(failed)}/{len(_checks)} checks passed")
    if failed:
        print("FAILED CHECKS:")
        for lbl in failed:
            print(f"  - {lbl}")
    print("=" * 70)
    return not failed


# RAG answer carrying all three real-run tell classes + protected medical content.
_RAG_WITH_TELLS = (
    "**Clinical Summary**\n"
    "A 55-year-old with type 2 diabetes, HbA1c 7.8%, on metformin 500 mg twice daily [1].\n\n"
    "**Evidence-Based Insights**\n"
    "According to the WHO classification provided in the Context, initial workup should "
    "include fasting glucose and a lipid panel.\n"
    "The Context does not provide specific treatment recommendations.\n\n"
    "**Limitations**\n"
    "Specialist endocrinology input may be required.\n\n"
    "This output is for informational use by medical professionals only and does not "
    "constitute a diagnosis or treatment recommendation."
)
# Vanilla answer: clean prose, NONE of the tells (so the rules must be a no-op here).
_VANILLA_CLEAN = (
    "For a 55-year-old with type 2 diabetes and HbA1c 7.8% on metformin 500 mg twice daily, "
    "check fasting glucose and a lipid panel, and consider endocrinology referral."
)


def run_tell_normalization():
    section("STEP 6: tell-class normalization (headers / footer / Context)")
    rec = {
        "item_id": "tell-001", "gold_specialty": "endocrinology",
        "available_specialties": ["cardiology", "endocrinology", "gastroenterology", "infectious_diseases"],
        "routed_specialty": "endocrinology",
        "case_ru": "[DUMMY] clinical case text",
        "answer_rag_ru": _RAG_WITH_TELLS,
        "answer_vanilla_ru": _VANILLA_CLEAN,
    }
    result = build_items.build([rec], mode="manual")
    tr = result.traces[0]
    rag_out = tr.rag.normalized_ru
    van_out = tr.vanilla.normalized_ru

    print("----- RAG normalized (served) -----")
    print(rag_out)
    print("----- RAG normalization diff -----")
    print(tr.rag.diff)
    print("----- RAG edits -----")
    print(json.dumps(tr.rag.edits, ensure_ascii=False))
    print("----- VANILLA normalized (served) -----")
    print(van_out)

    # (1) Templated headers removed; body under them kept.
    check("section header '**Clinical Summary**' removed", "**Clinical Summary**" not in rag_out)
    check("all three templated headers removed",
          not any(h in rag_out for h in ("**Clinical Summary**", "**Evidence-Based Insights**", "**Limitations**")))
    check("body under headers preserved",
          "fasting glucose and a lipid panel" in rag_out
          and "Specialist endocrinology input may be required." in rag_out)

    # (2) Boilerplate disclaimer footer removed.
    check("disclaimer footer removed",
          "informational use by medical professionals" not in rag_out)

    # (3) Context meta-reference neutralized; clinical sentence preserved.
    check("'provided in the Context' framing dropped", "provided in the Context" not in rag_out)
    check("clinical claim around the Context-ref preserved",
          "According to the WHO classification, initial workup should include fasting glucose and a lipid panel." in rag_out)
    check("declination (treatment not provided) survived",
          "does not provide specific treatment recommendations" in rag_out)
    check("no literal 'the Context' remains", "the Context" not in rag_out)

    # Content-preserving: numbers + drug names untouched.
    check("numbers/drug preserved (HbA1c 7.8%, metformin 500 mg)",
          "HbA1c 7.8%" in rag_out and "metformin 500 mg" in rag_out)

    # Meaning-sensitive edits captured + flagged for sign-off.
    ms = tr.rag.meaning_sensitive_edits
    print("  meaning-sensitive edits:", json.dumps(ms, ensure_ascii=False))
    check("meaning-sensitive Context edits recorded", len(ms) >= 1)
    check("review bundle flags meaning-sensitive edits for sign-off",
          "MEANING-SENSITIVE EDITS" in result.review_md
          and "REQUIRE EXPLICIT SIGN-OFF" in result.review_md)

    # No-op on clean text + identical rules applied to both arms.
    check("rules are a NO-OP on the clean vanilla answer", van_out == _VANILLA_CLEAN)
    check("vanilla had no edits (asymmetry removed, not created)", tr.vanilla.edits == [])
    check("SAME rule set applied to both arms", result.rules == build_items.DEFAULT_RULES)

    # Diffs emitted for both arms (RAG changed; vanilla unchanged).
    check("RAG diff emitted (non-empty change)", "before_normalization" in tr.rag.diff)
    check("vanilla diff is the explicit no-change marker", "no change" in tr.vanilla.diff)


_NO_COVERAGE_STUB = "Insufficient evidence in the current knowledge base to address this specific query."
_RAG_EN_WITH_TELLS = (
    "**Clinical Summary**\n"
    "A 58-year-old post-CABG with a pericardial friction rub; ESR/CRP elevated.\n\n"
    "**Evidence-Based Insights**\n"
    "According to the provided Context, NSAIDs are first-line for pericarditis (Source: 0002.txt).\n\n"
    "**Limitations**\n"
    "The provided Context does not provide specific guidance for this scenario.\n\n"
    "This output is for informational use by medical professionals only and does not "
    "constitute a diagnosis or treatment recommendation."
)
_VANILLA_EN_CLEAN = "Likely post-cardiac-injury (Dressler) syndrome; NSAIDs are reasonable."
_TELLS = ["**Clinical Summary**", "**Evidence-Based Insights**", "**Limitations**",
          "(Source:", "the Context", "provided Context",
          "informational use by medical professionals"]


def run_filter_and_ordering():
    section("STEP 7: no-coverage filter + normalize-BEFORE-translate (assisted)")
    captured = []  # every text handed to the (mock) translator

    def mock_translate(text):
        captured.append(text)
        return "RU«" + text + "»"   # mock RU: wraps so we can still inspect content

    recs = [
        {"item_id": "keep-1", "gold_specialty": "cardiology",
         "available_specialties": ["cardiology", "endocrinology", "gastroenterology", "infectious_diseases"],
         "routed_specialty": "cardiology", "case_en": "Post-CABG pericardial rub.",
         "answer_rag_en": _RAG_EN_WITH_TELLS, "answer_vanilla_en": _VANILLA_EN_CLEAN},
        {"item_id": "stub-1", "gold_specialty": "infectious_diseases",
         "available_specialties": ["cardiology", "endocrinology", "gastroenterology", "infectious_diseases"],
         "routed_specialty": "infectious_diseases", "case_en": "Some infection case.",
         "answer_rag_en": _NO_COVERAGE_STUB, "answer_vanilla_en": "Start empiric antibiotics."},
    ]
    result = build_items.build(recs, mode="assisted", translator=mock_translate,
                               exclude_no_coverage=True)

    # --- filter ---
    print("  excluded:", json.dumps([{"item_id": e["item_id"], "marker": e["matched_marker"][:30] + "..."}
                                      for e in result.excluded_no_coverage], ensure_ascii=False))
    print("  kept:", [it["item_id"] for it in result.items])
    check("no-coverage stub item excluded", [e["item_id"] for e in result.excluded_no_coverage] == ["stub-1"])
    check("excluded record logs the matched marker",
          result.excluded_no_coverage[0]["matched_marker"] == _NO_COVERAGE_STUB)
    check("substantive item kept", [it["item_id"] for it in result.items] == ["keep-1"])
    check("excluded item NOT in served items", all(it["item_id"] != "stub-1" for it in result.items))

    tr = result.traces[0]
    served_rag = result.items[0]["answer_rag_ru"]
    print("----- normalized EN (what was translated) -----")
    print(tr.rag.normalized_src)
    print("----- served RU (mock translation) -----")
    print(served_rag)

    # --- ordering: normalization ran BEFORE translation ---
    # The mock translator must have received TELL-FREE English (proves order).
    rag_input_to_translator = tr.rag.normalized_src
    check("translator received the normalized (tell-free) English",
          all(t not in rag_input_to_translator for t in _TELLS))
    check("every text passed to the translator is tell-free",
          all(all(t not in c for t in _TELLS) for c in captured))
    # And the served RU (post-translation) carries no tells either.
    check("served RU is tell-free", all(t not in served_rag for t in _TELLS))
    check("declination preserved through normalize+translate",
          "does not provide specific guidance" in served_rag)
    check("clinical claim preserved", "NSAIDs are first-line for pericarditis" in served_rag)

    # --- identical rules to both arms; vanilla (clean) is a no-op ---
    check("vanilla had no edits (no-op; asymmetry removed not created)", tr.vanilla.edits == [])
    check("vanilla normalized == vanilla source (unchanged)", tr.vanilla.normalized_src == _VANILLA_EN_CLEAN)
    check("SAME rule set applied to both arms", result.rules == build_items.DEFAULT_RULES)

    # --- Context edits are meaning-sensitive, diffed + flagged ---
    ms = tr.rag.meaning_sensitive_edits
    print("  meaning-sensitive edits:", json.dumps(ms, ensure_ascii=False))
    check("Context edits recorded as meaning-sensitive", len(ms) >= 1)
    check("review bundle flags meaning-sensitive edits + lists the exclusion",
          "MEANING-SENSITIVE EDITS" in result.review_md and "stub-1" in result.review_md)
    check("RAG normalization diff emitted", "before_normalization" in tr.rag.norm_diff)


def test_smoke_items_build():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
