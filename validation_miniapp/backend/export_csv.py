import argparse
import csv
import io
from pathlib import Path

from . import db

_PREF_TOWARD_OPT1 = {
    "opt1_strong": 2,
    "opt1_weak": 1,
    "tie": 0,
    "opt2_weak": -1,
    "opt2_strong": -2,
}

CSV_COLUMNS = [
    "rater_id",          # hashed
    "item_id",
    "gold_specialty",
    "routed_specialty",
    "routing_judgment",
    "pref_toward_rag",   # +2..-2, + means RAG arm preferred
    "safety_flag_rag",
    "safety_flag_vanilla",
    "client_ts",
]


def _pref_toward_rag(preference, option_1_arm):
    """Signed preference toward the RAG arm in {+2,+1,0,-1,-2}.

    ``preference`` is positional. We score it toward option_1, then flip the sign
    if option_1 is the *vanilla* arm (so option_2 is RAG).
    """
    toward_opt1 = _PREF_TOWARD_OPT1[preference]
    if option_1_arm == "rag":
        return toward_opt1
    # option_1 is vanilla => positive-toward-opt1 means negative-toward-rag.
    return -toward_opt1


def rows_for_export(conn, items_by_id):
    """Yield de-blinded dict rows, one per submission.

    ``items_by_id`` maps item_id -> item dict (for gold/routed specialty lookup).
    """
    out = []
    for sub in db.all_submissions(conn):
        rater_id = sub["rater_id"]
        item_id = sub["item_id"]
        pos = db.get_position(conn, rater_id, item_id)
        if pos is None:
            raise RuntimeError(
                f"submission ({rater_id}, {item_id}) has no position map row"
            )
        opt1_arm = pos["option_1_arm"]

        if opt1_arm == "rag":
            safety_rag, safety_vanilla = sub["safety_flag_opt1"], sub["safety_flag_opt2"]
        else:
            safety_rag, safety_vanilla = sub["safety_flag_opt2"], sub["safety_flag_opt1"]

        item = items_by_id.get(item_id, {})
        out.append({
            "rater_id": rater_id,
            "item_id": item_id,
            "gold_specialty": item.get("gold_specialty", ""),
            "routed_specialty": item.get("routed_specialty", ""),
            "routing_judgment": sub["routing_judgment"],
            "pref_toward_rag": _pref_toward_rag(sub["preference"], opt1_arm),
            "safety_flag_rag": int(safety_rag),
            "safety_flag_vanilla": int(safety_vanilla),
            "client_ts": sub["client_ts"] if sub["client_ts"] is not None else "",
        })
    return out


def to_csv_string(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def export(conn, items_by_id, out_path=None):
    """Build the CSV string; optionally write it to ``out_path``. Returns the string."""
    rows = rows_for_export(conn, items_by_id)
    text = to_csv_string(rows)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return text


def _load_items(items_path):
    import json
    data = json.loads(Path(items_path).read_text(encoding="utf-8"))
    return {it["item_id"]: it for it in data}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export de-blinded judgments to CSV.")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    parser.add_argument(
        "--items",
        default=str(Path(__file__).resolve().parent.parent / "data" / "items.dummy.json"),
    )
    parser.add_argument("--out", default=None, help="Output CSV path; prints to stdout if omitted.")
    args = parser.parse_args(argv)

    conn = db.connect(args.db)
    items_by_id = _load_items(args.items)
    text = export(conn, items_by_id, out_path=args.out)
    if args.out:
        print(f"Wrote {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
