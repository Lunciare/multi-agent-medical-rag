"""Run the golden cases through both study arms and emit raw inputs for build_items.py.

For each golden case this runs:
  - the RAG arm  = the team's REAL pipeline, via ``MedicalOrchestrator(kb).answer(case)``
                   (orchestrator -> specialist, with retrieval). We capture the answer,
                   the orchestrator's emitted routed specialist, and the retrieved context.
  - the vanilla arm = the SAME base model answering the raw case with a minimal, neutral
                   system prompt, no routing, no retrieval.

It does NOT reimplement or modify the pipeline — it only calls it (thin adapter). The two
arms differ ONLY in pipeline-vs-bare: same base model + decoding params.

INTEGRITY (enforced below):
  - Each case is run exactly ONCE per arm under a fixed, logged config. Genuine transport
    errors may be retried (``--retries``) and every attempt is logged. We never generate
    multiple candidates and pick one.
  - Answers are saved VERBATIM, including empty/error outputs. Failures are recorded (in the
    per-case trace and a failures summary); a case is never silently dropped.
  - The routed specialist is captured exactly as the orchestrator emits it — never inferred.
  - We use ONLY the case text + specialty label from the dataset; never its gold answer.

The real arms are built lazily (see ``build_real_arms``) so this module imports offline and
the smoke test can inject mock arms without touching the pipeline or any API.

CLI::

    python tools/run_cases.py --config tools/run_cases.config.example.json \
        --limit 5            # small real subset first (cost/correctness check)
    python tools/run_cases.py --config tools/run_cases.config.example.json   # full run
"""

import argparse
import json
import logging
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("run_cases")

_VALIDATION_ROOT = Path(__file__).resolve().parent.parent          # validation_miniapp/
_REPO_ROOT = _VALIDATION_ROOT.parent                                # repo root
_MAS_DIR = _REPO_ROOT / "multi-agent_system"

# Single source of truth for specialty vocabulary (dataset + emitted -> canonical).
if str(_VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_ROOT))
from backend import specialties  # noqa: E402

# Consolidated raw file must match tools/build_items.py's assisted-mode schema EXACTLY.
RAW_FIELDS = ("item_id", "gold_specialty", "available_specialties", "routed_specialty",
              "case_en", "answer_rag_en", "answer_vanilla_en")

# Pre-specified, reproducible balanced sample: N per specialist -> 4 x N cases.
DEFAULT_PER_SPECIALTY = 3        # 3 x 4 = 12-case rater session
DEFAULT_SEED = 20260531          # fixed BEFORE seeing any model outputs


def _now():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    golden_path: str
    vanilla_system_prompt: str
    decoding: dict                     # {temperature, max_tokens, seed}
    selection: dict = field(default_factory=dict)   # {per_specialty, seed, ids}
    knowledge_base_dir: str | None = None
    out_root: str = str(_VALIDATION_ROOT / "results")
    retries: int = 0
    raw: dict = field(default_factory=dict)   # the full config dict, logged verbatim

    @staticmethod
    def load(path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        required = ("golden_path", "vanilla_system_prompt", "decoding")
        for k in required:
            if k not in data:
                raise ValueError(f"config missing required key: {k!r}")
        return Config(
            golden_path=data["golden_path"],
            vanilla_system_prompt=data["vanilla_system_prompt"],
            decoding=dict(data["decoding"]),
            selection=dict(data.get("selection", {})),
            knowledge_base_dir=data.get("knowledge_base_dir"),
            out_root=data.get("out_root", str(_VALIDATION_ROOT / "results")),
            retries=int(data.get("retries", 0)),
            raw=data,
        )


# --------------------------------------------------------------------------- #
# Golden dataset -> our case schema                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    item_id: str
    gold_specialty: str
    case_en: str
    source_record: dict


def load_golden(golden_path):
    """Map ALL golden records -> Case list over the four canonical specialties.

    Uses ONLY id, query, and expected_specialist. ``expected_specialist`` is mapped to a
    canonical specialty via backend.specialties (so dataset + emitted labels agree). Raises
    on missing fields or an unrecognised specialist label (we never invent a label).
    Returns (cases, flags).
    """
    records = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"golden dataset must be a non-empty JSON array: {golden_path}")

    cases = []
    flags = []
    seen = set()
    per_specialty = {}
    for i, rec in enumerate(records):
        rid = rec.get("id")
        spec = rec.get("expected_specialist")
        query = rec.get("query")
        if not rid:
            raise ValueError(f"golden[{i}]: missing 'id'")
        if not spec:
            raise ValueError(f"golden id={rid!r}: missing 'expected_specialist' (need it "
                             f"for routing accuracy — not inventing a label)")
        if not query or not str(query).strip():
            raise ValueError(f"golden id={rid!r}: missing/empty 'query'")
        canonical = specialties.to_canonical(spec)
        if canonical is None:
            raise ValueError(f"golden id={rid!r}: expected_specialist {spec!r} does not map "
                             f"to a canonical specialty {specialties.CANONICAL_SPECIALTIES}")
        if rid in seen:
            raise ValueError(f"golden id={rid!r}: duplicate id")
        seen.add(rid)
        per_specialty[canonical] = per_specialty.get(canonical, 0) + 1
        cases.append(Case(item_id=rid, gold_specialty=canonical,
                          case_en=str(query), source_record=rec))
    flags.append(f"loaded {len(cases)} cases across {len(per_specialty)} canonical "
                 f"specialties: {per_specialty}")
    return cases, flags


def select_balanced(cases, *, per_specialty=DEFAULT_PER_SPECIALTY, seed=DEFAULT_SEED, ids=None):
    """Pick a balanced, reproducible sample for the rater session.

    Default: ``per_specialty`` cases per canonical specialty (3 x 4 = 12), randomly sampled
    within each specialty under a FIXED ``seed`` so the same seed always yields the same ids.
    The rule is fixed BEFORE seeing any model outputs; selection uses only ids + the gold
    specialty, never any answer. If ``ids`` is given, it overrides sampling (explicit set).
    Returns (selected_cases, selection_meta).
    """
    by_id = {c.item_id: c for c in cases}

    if ids:
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"explicit selection ids not found in golden: {missing}")
        selected = [by_id[i] for i in ids]
        meta = {"mode": "explicit_ids", "ids": list(ids),
                "selected_ids": [c.item_id for c in selected],
                "per_specialty_counts": _count_by_specialty(selected)}
        return selected, meta

    rng = random.Random(seed)
    groups = {}
    for c in cases:
        groups.setdefault(c.gold_specialty, []).append(c)

    selected = []
    shortfalls = {}
    # Deterministic order over canonical specialties; sample within each from sorted ids.
    for spec in specialties.CANONICAL_SPECIALTIES:
        pool = sorted(groups.get(spec, []), key=lambda c: c.item_id)
        if len(pool) < per_specialty:
            shortfalls[spec] = {"have": len(pool), "need": per_specialty}
            chosen = pool
        else:
            chosen = rng.sample(pool, per_specialty)
        chosen = sorted(chosen, key=lambda c: c.item_id)
        selected.extend(chosen)

    meta = {
        "mode": "balanced_sample",
        "per_specialty": per_specialty,
        "seed": seed,
        "selected_ids": [c.item_id for c in selected],
        "per_specialty_counts": _count_by_specialty(selected),
    }
    if shortfalls:
        meta["shortfalls"] = shortfalls
    return selected, meta


def _count_by_specialty(cases):
    out = {}
    for c in cases:
        out[c.gold_specialty] = out.get(c.gold_specialty, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# Arm invocation (single run, log errors, never drop)                          #
# --------------------------------------------------------------------------- #
def _invoke_arm(name, fn, case_text, retries):
    """Call an arm callable once (with optional retries on exception). Always returns a
    dict with at least 'answer' and 'error'; never raises. Logs every attempt."""
    attempt = 0
    last_err = None
    started = _now()
    while attempt <= retries:
        attempt += 1
        try:
            res = dict(fn(case_text))
            res.setdefault("answer", "")
            res.setdefault("error", None)
            res["attempts"] = attempt
            res["started_at"] = started
            res["ended_at"] = _now()
            if not str(res.get("answer", "")).strip():
                logger.warning("arm=%s produced EMPTY answer (recorded verbatim)", name)
            return res
        except Exception as e:  # genuine error: log, maybe retry
            last_err = f"{type(e).__name__}: {e}"
            logger.error("arm=%s attempt=%d failed: %s", name, attempt, last_err)
    return {"answer": "", "error": last_err, "attempts": attempt,
            "started_at": started, "ended_at": _now()}


def run_cases(cases, *, rag_arm, vanilla_arm, available_specialties, config,
              run_dir, run_meta, limit=None):
    """Run both arms over ``cases``, writing per-case traces + consolidated raw file.

    ``rag_arm``/``vanilla_arm`` are injectable callables (real pipeline, or mocks in tests).
    Pure w.r.t. the pipeline: it never imports it. Returns a summary dict.
    """
    run_dir = Path(run_dir)
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    selected = cases if limit is None else cases[:limit]
    raw_records = []
    failures = []

    for n, case in enumerate(selected, 1):
        logger.info("[%d/%d] %s (gold=%s)", n, len(selected), case.item_id, case.gold_specialty)
        rag = _invoke_arm("rag", rag_arm, case.case_en, config.retries)
        van = _invoke_arm("vanilla", vanilla_arm, case.case_en, config.retries)

        # Map the EMITTED routed label to canonical for storage/comparison, but keep the raw
        # emitted value in the trace for audit. Unmappable emissions (e.g. the orchestrator's
        # "Error"/"Safety Gateway" short-circuits) are preserved verbatim, never fabricated.
        routed_raw = rag.get("routed_specialty", "")
        routed_canonical = specialties.to_canonical(routed_raw)
        rag["routed_specialty_raw"] = routed_raw
        rag["routed_specialty_canonical"] = routed_canonical
        stored_routed = routed_canonical if routed_canonical is not None else routed_raw

        # Consolidated raw record — VERBATIM answers, schema EXACTLY build_items assisted-mode.
        # answer_rag_en derives ONLY from the RAG arm; answer_vanilla_en ONLY from vanilla.
        raw_records.append({
            "item_id": case.item_id,
            "gold_specialty": case.gold_specialty,
            "available_specialties": list(available_specialties),
            "routed_specialty": stored_routed,
            "case_en": case.case_en,
            "answer_rag_en": rag.get("answer", ""),
            "answer_vanilla_en": van.get("answer", ""),
        })

        trace = {
            "item_id": case.item_id,
            "gold_specialty": case.gold_specialty,
            "available_specialties": list(available_specialties),
            "case_en": case.case_en,
            "rag_arm": rag,
            "vanilla_arm": van,
            "source_record": case.source_record,
            "run_id": run_meta["run_id"],
            "written_at": _now(),
        }
        (traces_dir / f"{case.item_id}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

        # Record (don't drop) failures: arm error OR empty answer.
        problems = []
        for arm_name, res in (("rag", rag), ("vanilla", van)):
            if res.get("error"):
                problems.append(f"{arm_name}:error:{res['error']}")
            elif not str(res.get("answer", "")).strip():
                problems.append(f"{arm_name}:empty_answer")
        if problems:
            failures.append({"item_id": case.item_id, "problems": problems})

    raw_path = run_dir / "items_raw.json"
    raw_path.write_text(json.dumps(raw_records, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    failures_path = run_dir / "failures.json"
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    summary = {
        "run_id": run_meta["run_id"],
        "run_dir": str(run_dir),
        "cases_total": len(cases),
        "cases_run": len(selected),
        "failures": len(failures),
        "raw_inputs_file": str(raw_path),
        "failures_file": str(failures_path),
    }
    manifest = {**run_meta, "config": config.raw,
                "available_specialties": list(available_specialties),
                "summary": summary, "flags": run_meta.get("flags", [])}
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# Real arms (lazy import of the pipeline — only for actual runs)               #
# --------------------------------------------------------------------------- #
def build_real_arms(config):
    """Construct the REAL rag/vanilla arms by importing multi-agent_system. Lazy on purpose:
    importing settings requires YANDEX_* env, and the pipeline loads FAISS indexes. Returns
    (rag_arm, vanilla_arm, available_specialties, arm_meta)."""
    if str(_MAS_DIR) not in sys.path:
        sys.path.insert(0, str(_MAS_DIR))
    import settings  # noqa: E402  (raises if YANDEX_* unset — that's the real prereq)
    from orchestrator import MedicalOrchestrator  # noqa: E402

    kb = config.knowledge_base_dir or settings.DEFAULT_KNOWLEDGE_BASE_DIR
    orch = MedicalOrchestrator(kb)
    # The real router targets, mapped to canonical for storage; raw kept in arm_meta.
    allowed_raw = list(orch.allowed_specialists)
    available = specialties.available_canonical(allowed_raw)
    dec = config.decoding

    def rag_arm(case_text):
        # The REAL pipeline: orchestrator routes + the specialist retrieves + answers.
        specialist, response, evidence = orch.answer(case_text)
        return {
            "answer": response,                  # verbatim
            "routed_specialty": specialist,      # as emitted, never overridden
            "retrieved_context": evidence,       # retrieved chunks markdown
            "model_id": settings.AGENT_MODEL,
            "config": {
                "kind": "rag_pipeline",
                "routing_model": settings.ROUTING_MODEL,
                "agent_model": settings.AGENT_MODEL,
                # Decoding is fixed inside the pipeline (specialist.py): temperature=0.0,
                # max_tokens=1024. Recorded here as observed (this runner does not set it).
                "rag_decoding_observed": {"temperature": 0.0, "max_tokens": 1024},
                "retrieval_observed": {
                    "similarity_top_k": settings.SIMILARITY_TOP_K,
                    "max_l2_distance": settings.MAX_L2_DISTANCE,
                },
                "knowledge_base_dir": kb,
            },
        }

    def vanilla_arm(case_text):
        # Bare base model: same client/model as the specialist generation call, no routing,
        # no retrieval, minimal neutral system prompt. Same decoding params.
        resp = settings.client.chat.completions.create(
            model=settings.AGENT_MODEL,
            messages=[
                {"role": "system", "content": config.vanilla_system_prompt},
                {"role": "user", "content": case_text},
            ],
            temperature=dec.get("temperature", 0.0),
            max_tokens=dec.get("max_tokens", 1024),
            extra_headers={"x-folder-id": settings.YANDEX_PROJECT_ID},
            **({"seed": dec["seed"]} if dec.get("seed") is not None else {}),
        )
        return {
            "answer": resp.choices[0].message.content.strip(),
            "model_id": settings.AGENT_MODEL,
            "config": {
                "kind": "vanilla_bare_model",
                "agent_model": settings.AGENT_MODEL,
                "decoding": {"temperature": dec.get("temperature", 0.0),
                             "max_tokens": dec.get("max_tokens", 1024),
                             "seed": dec.get("seed")},
                "system_prompt": config.vanilla_system_prompt,
                "retrieval": "none", "routing": "none",
            },
        }

    arm_meta = {
        "agent_model": settings.AGENT_MODEL,
        "routing_model": settings.ROUTING_MODEL,
        "embedding_model": getattr(settings, "EMBEDDING_MODEL", None),
        "allowed_specialists_raw": allowed_raw,
        "available_specialties_canonical": available,
    }
    return rag_arm, vanilla_arm, available, arm_meta


# --------------------------------------------------------------------------- #
# Run dir + git metadata                                                       #
# --------------------------------------------------------------------------- #
def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT),
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def make_run_dir(out_root):
    """Create an isolated, enumerated run dir: results/run_<NNNN>_<UTC>/ ."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in out_root.glob("run_*") if p.is_dir()]
    idx = 1 + max([int(n.split("_")[1]) for n in existing
                   if n.split("_")[1:2] and n.split("_")[1].isdigit()] or [0])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"run_{idx:04d}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Run golden cases through RAG + vanilla arms.")
    p.add_argument("--config", required=True)
    p.add_argument("--golden", default=None, help="override golden_path from config")
    p.add_argument("--limit", type=int, default=None, help="further cap: run only first N selected")
    p.add_argument("--per-specialty", type=int, default=None, help="override selection.per_specialty")
    p.add_argument("--seed", type=int, default=None, help="override selection.seed")
    p.add_argument("--out-root", default=None, help="override results root")
    p.add_argument("--retries", type=int, default=None, help="retries per arm on error")
    args = p.parse_args(argv)

    config = Config.load(args.config)
    if args.golden:
        config.golden_path = args.golden
    if args.out_root:
        config.out_root = args.out_root
    if args.retries is not None:
        config.retries = args.retries

    cases, flags = load_golden(config.golden_path)
    for fl in flags:
        logger.info("FLAG: %s", fl)

    # Balanced, reproducible selection (fixed seed; rule set before seeing outputs).
    sel = config.selection
    per_specialty = args.per_specialty if args.per_specialty is not None \
        else sel.get("per_specialty", DEFAULT_PER_SPECIALTY)
    seed = args.seed if args.seed is not None else sel.get("seed", DEFAULT_SEED)
    selected, selection_meta = select_balanced(
        cases, per_specialty=per_specialty, seed=seed, ids=sel.get("ids"))
    logger.info("Selected %d cases: %s", len(selected), selection_meta)

    logger.info("Building real arms (lazy import of multi-agent_system)...")
    rag_arm, vanilla_arm, available, arm_meta = build_real_arms(config)

    run_dir = make_run_dir(config.out_root)
    started = _now()
    run_meta = {
        "run_id": run_dir.name,
        "started_at": started,
        "git_commit": _git_commit(),
        "golden_path": config.golden_path,
        "arm_meta": arm_meta,
        "selection": selection_meta,
        "flags": flags,
        "integrity": {
            "runs_per_arm_per_case": 1,
            "retries": config.retries,
            "answers_saved_verbatim": True,
            "routed_specialty_source": "orchestrator.answer() emitted value -> canonical; raw kept in trace",
            "selection_rule_fixed_before_outputs": True,
            "dataset_fields_used": ["id", "query", "expected_specialist"],
        },
    }
    summary = run_cases(selected, rag_arm=rag_arm, vanilla_arm=vanilla_arm,
                        available_specialties=available, config=config,
                        run_dir=run_dir, run_meta=run_meta, limit=args.limit)
    summary["ended_at"] = _now()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nNEXT: feed {summary['raw_inputs_file']} to tools/build_items.py "
          f"(--mode assisted) and run the human review before go-live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
