import json
import sys
import tempfile
from pathlib import Path

_MINIAPP = Path(__file__).resolve().parent.parent
if str(_MINIAPP) not in sys.path:
    sys.path.insert(0, str(_MINIAPP))

from backend import specialties  # noqa: E402
from tools import run_cases  # noqa: E402

FIXTURE = _MINIAPP / "tests" / "fixtures" / "golden.dummy.json"
MOCK_AVAILABLE = ["cardiology", "endocrinology", "gastroenterology", "infectious_diseases"]
MOCK_EMITTED_ROUTED = "Infectionist"   # exercises the trickiest canonical mapping

_checks = []


def check(label, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    _checks.append((label, bool(cond)))
    return bool(cond)


def section(t):
    print("\n" + "=" * 70); print(t); print("=" * 70)


# --- mock arms -------------------------------------------------------------- #
def mock_rag_arm(case_text):
    return {
        "answer": f"RAG_STUB_ANSWER for <{case_text[:30]}>",
        "routed_specialty": MOCK_EMITTED_ROUTED,   # RAW emitted label (display form)
        "retrieved_context": "STUB retrieved chunk [Source: dummy.txt]",
        "model_id": "mock-model",
        "config": {"kind": "rag_pipeline_mock"},
    }


def mock_vanilla_arm(case_text):
    if "TRIGGER_VANILLA_FAILURE" in case_text:
        raise RuntimeError("simulated vanilla API failure")
    return {
        "answer": f"VANILLA_STUB_ANSWER for <{case_text[:30]}>",
        "model_id": "mock-model",
        "config": {"kind": "vanilla_bare_model_mock"},
    }


def run():
    cfg = run_cases.Config.load(str(_MINIAPP / "tools" / "run_cases.config.example.json"))

    section("STEP 1: load_golden maps ALL FOUR specialists to canonical (no filter)")
    cases, flags = run_cases.load_golden(str(FIXTURE))
    counts = run_cases._count_by_specialty(cases)
    print(f"  loaded {len(cases)} cases; per-specialty: {counts}")
    print(f"  flags: {flags}")
    check("all 16 dummy cases loaded (no 2-specialty filter)", len(cases) == 16)
    check("four canonical specialties present",
          set(counts) == set(specialties.CANONICAL_SPECIALTIES))
    check("balanced fixture: 4 per specialty", set(counts.values()) == {4})

    section("STEP 2: balanced selection = 3 per specialty (12), reproducible under seed")
    sel1, meta1 = run_cases.select_balanced(cases, per_specialty=3, seed=20260531)
    sel2, meta2 = run_cases.select_balanced(cases, per_specialty=3, seed=20260531)
    sel_other, meta_other = run_cases.select_balanced(cases, per_specialty=3, seed=999)
    print(f"  selected ids (seed 20260531): {meta1['selected_ids']}")
    print(f"  per-specialty counts: {meta1['per_specialty_counts']}")
    print(f"  selected ids (seed 999):      {meta_other['selected_ids']}")
    check("selection yields exactly 12", len(sel1) == 12)
    check("selection is balanced 3 x 4", meta1["per_specialty_counts"] == {s: 3 for s in specialties.CANONICAL_SPECIALTIES})
    check("same seed -> identical ids (reproducible)", meta1["selected_ids"] == meta2["selected_ids"])
    check("different seed -> (generally) different ids", meta_other["selected_ids"] != meta1["selected_ids"])

    section("STEP 3: run_cases on an explicit selection incl. a failing case (mock arms)")
    # Explicit ids: one per specialty, including the trigger case, so the run is deterministic.
    chosen_ids = ["dummy_cardio_1", "dummy_endo_2", "dummy_gastro_1", "dummy_infect_1"]
    selected, selection_meta = run_cases.select_balanced(cases, ids=chosen_ids)
    out_root = Path(tempfile.mkdtemp()) / "results"
    run_dir = run_cases.make_run_dir(out_root)
    run_meta = {"run_id": run_dir.name, "started_at": run_cases._now(),
                "git_commit": "test", "selection": selection_meta, "flags": flags}
    summary = run_cases.run_cases(
        selected, rag_arm=mock_rag_arm, vanilla_arm=mock_vanilla_arm,
        available_specialties=MOCK_AVAILABLE, config=cfg, run_dir=run_dir, run_meta=run_meta)
    print("  summary:", json.dumps(summary, ensure_ascii=False))
    check("ran all 4 chosen cases", summary["cases_run"] == 4)

    raw = json.loads((run_dir / "items_raw.json").read_text(encoding="utf-8"))
    print("\n----- items_raw.json BEGIN -----")
    print(json.dumps(raw, ensure_ascii=False, indent=2))
    print("----- items_raw.json END -----")
    raw_by_id = {r["item_id"]: r for r in raw}

    section("STEP 4: no-swap + emitted->canonical routing (raw kept in trace)")
    canon_expected = specialties.to_canonical(MOCK_EMITTED_ROUTED)
    print(f"  emitted routed = {MOCK_EMITTED_ROUTED!r} -> canonical {canon_expected!r}")
    for r in raw:
        rag_ok = "RAG_STUB_ANSWER" in r["answer_rag_en"] and "VANILLA_STUB" not in r["answer_rag_en"]
        van_ok = ("VANILLA_STUB_ANSWER" in r["answer_vanilla_en"]
                  and "RAG_STUB" not in r["answer_vanilla_en"]) or r["answer_vanilla_en"] == ""
        routed_ok = r["routed_specialty"] == canon_expected
        avail_ok = r["available_specialties"] == MOCK_AVAILABLE
        print(f"  {r['item_id']}: rag_ok={rag_ok} van_ok={van_ok} "
              f"routed={r['routed_specialty']!r} avail_canonical={avail_ok}")
        check(f"{r['item_id']}: answer_rag_en from RAG stub only", rag_ok)
        check(f"{r['item_id']}: answer_vanilla_en from vanilla stub only (or empty if failed)", van_ok)
        check(f"{r['item_id']}: routed_specialty stored as canonical", routed_ok)
        check(f"{r['item_id']}: available_specialties are canonical (4)", avail_ok)
    # RAW emitted value preserved in the trace for audit.
    t0 = json.loads((run_dir / "traces" / "dummy_cardio_1.json").read_text(encoding="utf-8"))
    print("\n  trace rag_arm.routed_specialty_raw:", t0["rag_arm"]["routed_specialty_raw"])
    print("  trace rag_arm.routed_specialty_canonical:", t0["rag_arm"]["routed_specialty_canonical"])
    check("trace keeps RAW emitted routing label", t0["rag_arm"]["routed_specialty_raw"] == MOCK_EMITTED_ROUTED)
    check("trace records canonical routing label", t0["rag_arm"]["routed_specialty_canonical"] == canon_expected)

    section("STEP 5: simulated failure recorded WITHOUT dropping the case")
    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
    print("  failures.json:", json.dumps(failures, ensure_ascii=False))
    failed_ids = {f["item_id"] for f in failures}
    check("triggered case in failures.json", "dummy_endo_2" in failed_ids)
    check("failed case NOT dropped (still in raw)", "dummy_endo_2" in raw_by_id)
    check("failed case's vanilla answer saved verbatim as empty",
          raw_by_id["dummy_endo_2"]["answer_vanilla_en"] == "")
    check("failed case's RAG answer still captured",
          "RAG_STUB_ANSWER" in raw_by_id["dummy_endo_2"]["answer_rag_en"])

    section("STEP 6: run_manifest.json logs the selected ids + canonical available set")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    print("  manifest.selection.selected_ids:", manifest["selection"]["selected_ids"])
    print("  manifest.available_specialties:", manifest["available_specialties"])
    check("manifest logs selected ids", manifest["selection"]["selected_ids"] == chosen_ids)
    check("manifest records canonical available_specialties",
          manifest["available_specialties"] == MOCK_AVAILABLE)

    section("RESULT")
    failed = [lbl for lbl, ok in _checks if not ok]
    print(f"{len(_checks) - len(failed)}/{len(_checks)} checks passed")
    if failed:
        print("FAILED CHECKS:")
        for lbl in failed:
            print(f"  - {lbl}")
    print("=" * 70)
    return not failed


def test_smoke_run_cases():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
