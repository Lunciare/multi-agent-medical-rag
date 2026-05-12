#!/usr/bin/env python3
from pathlib import Path

PROCESSED_ROOTS = [
    Path("data/processed/cardiology"),
    Path("data/processed/endocrinology"),
]


def fix_summary(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.rstrip("\n").split("\n")

    kw_idx = None
    for i, line in enumerate(lines):
        if line.startswith("KEYWORDS:"):
            kw_idx = i
            break

    if kw_idx is None:
        return False

    title_block = "\n".join(lines[:kw_idx]).strip()
    kw_line = lines[kw_idx]

    after_kw = lines[kw_idx + 1:]

    blocks = []
    current = []
    for line in after_kw:
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())

    if len(blocks) < 2:
        return False

    first = blocks[0].strip()
    second = blocks[1].strip()

    second_clean = second.replace(". Brief note: source text is empty.", "").strip()

    if first == second_clean or first.startswith(second_clean[:200]) or second_clean.startswith(first[:200]):
        fixed = f"{title_block}\n{kw_line}\n\n{first}\n"
        path.write_text(fixed, encoding="utf-8")
        return True

    return False


def main():
    total = 0
    fixed = 0

    for root in PROCESSED_ROOTS:
        if not root.exists():
            print(f"[skip] {root} not found")
            continue

        for summary_path in sorted(root.rglob("summary.txt")):
            total += 1
            if fix_summary(summary_path):
                fixed += 1
                print(f"  [fixed] {summary_path.relative_to(root)}")

    print(f"\nDone. Scanned {total} summaries, fixed {fixed} duplicates.")


if __name__ == "__main__":
    main()
