#!/usr/bin/env python3
import re
import os
import shutil
from pathlib import Path

IN_ROOT = Path("data/raw/Endocrinology")
OUT_ROOT = Path("data/processed/endocrinology")
CATEGORIES = ["Articles", "Cases", "Guidelines", "Handbooks", "Textbooks"]

CHUNK_WORDS = 400
OVERLAP_WORDS = 30


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\/\\:\*\?"<>\|]', '_', name)
    name = re.sub(r'\s+', ' ', name)
    return name[:120] if len(name) > 120 else name


def normalize_text(text: str) -> str:
    text = text.replace("﻿", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def split_words(text: str):
    return re.findall(r"\S+", text)


def make_chunks(words, chunk_words: int, overlap_words: int):
    chunks = []
    start = 0
    n = len(words)
    step = chunk_words - overlap_words

    while start < n:
        end = min(start + chunk_words, n)
        chunks.append(words[start:end])
        if end == n:
            break
        start += step

    return chunks


def write_chunks(out_dir: Path, title: str, body: str):
    words = split_words(body) if body else []

    if len(words) < 10:
        return 0

    if len(words) <= CHUNK_WORDS:
        chunks = [words]
    else:
        chunks = make_chunks(words, CHUNK_WORDS, OVERLAP_WORDS)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk_words_list in enumerate(chunks, start=1):
        chunk_text = " ".join(chunk_words_list).strip()
        content = f"{title}\n\n{chunk_text}\n"
        out_path = out_dir / f"{i:04d}.txt"
        out_path.write_text(content, encoding="utf-8")

    return len(chunks)


def parse_ncbi_chapters(text: str):
    chapters = []
    parts = re.split(r'^--- CHAPTER: (.+?) ---$', text, flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        content = re.sub(r'^SOURCE:\s*\S+\s*', '', content, count=1).strip()
        if content:
            chapters.append((title, content))
    return chapters


def parse_endotext_chapters(text: str):
    chapters = []
    delim = "════════════════════════════════════════════════════════════"
    parts = text.split(delim)

    current_title = None
    current_content_lines = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.split("\n")
        chapter_match = None
        for line in lines:
            line = line.strip()
            if line.startswith("CHAPTER:"):
                chapter_match = line.replace("CHAPTER:", "").strip()
                break

        if chapter_match:
            if current_title and current_content_lines:
                content = "\n".join(current_content_lines).strip()
                if content:
                    chapters.append((current_title, content))
            current_title = chapter_match
            current_content_lines = []
        elif part.startswith("ENDOTEXT") or part.startswith("Extracted for RAG"):
            continue
        else:
            if current_title:
                current_content_lines.append(part)
            else:
                if part.startswith("INTRODUCTION"):
                    current_title = "Introduction"
                    current_content_lines.append(part)

    if current_title and current_content_lines:
        content = "\n".join(current_content_lines).strip()
        if content:
            chapters.append((current_title, content))

    return chapters


def parse_diabetes_chapters(text: str):
    chapters = []
    delim = "════════════════════════════════════════════════════════════"
    parts = text.split(delim)

    current_title = None
    current_content_lines = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.split("\n")
        chapter_match = None
        source_line = None

        for line in lines:
            line = line.strip()
            if line.startswith("CHAPTER:"):
                chapter_match = line.replace("CHAPTER:", "").strip()
            elif line.startswith("SOURCE:"):
                source_line = line

        if chapter_match:
            if current_title and current_content_lines:
                content = "\n".join(current_content_lines).strip()
                if content and current_title not in ("Cover", "Preface", "Update History"):
                    chapters.append((current_title, content))
            current_title = chapter_match
            current_content_lines = []
        elif part.startswith("DIABETES AMERICA") or part.startswith("Extracted for RAG"):
            continue
        else:
            if current_title:
                current_content_lines.append(part)

    if current_title and current_content_lines:
        content = "\n".join(current_content_lines).strip()
        if content and current_title not in ("Cover", "Preface", "Update History"):
            chapters.append((current_title, content))

    return chapters


def process_textbook(filepath: Path, out_base: Path):
    text = filepath.read_text(encoding="utf-8", errors="replace")
    text = normalize_text(text)
    name = filepath.stem

    if "NCBI" in name:
        chapters = parse_ncbi_chapters(text)
    elif "Endotext" in name:
        chapters = parse_endotext_chapters(text)
    elif "Diabetes" in name:
        chapters = parse_diabetes_chapters(text)
    else:
        title = name
        chapters = [(title, text)]

    total_chunks = 0
    for ch_title, ch_content in chapters:
        folder_name = sanitize_name(f"{name} - {ch_title}")
        out_dir = out_base / folder_name
        n = write_chunks(out_dir, ch_title, ch_content)
        total_chunks += n

    return len(chapters), total_chunks


def process_single_file(filepath: Path, out_base: Path):
    text = filepath.read_text(encoding="utf-8", errors="replace")
    text = normalize_text(text)

    lines = text.split("\n")
    title = (lines[0] or "").strip()
    if not title:
        title = filepath.stem

    body = "\n".join(lines[1:]).strip()
    folder_name = sanitize_name(filepath.stem)
    out_dir = out_base / folder_name

    n = write_chunks(out_dir, title, body)
    return n


def main():
    stats = {"docs": 0, "chapters": 0, "chunks": 0, "skipped": 0}

    for cat in CATEGORIES:
        in_dir = IN_ROOT / cat
        out_dir = OUT_ROOT / cat

        if not in_dir.exists():
            print(f"[skip] no folder: {in_dir}")
            continue

        txt_files = sorted([
            p for p in in_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        ])

        if not txt_files:
            print(f"[skip] no .txt files in {in_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {cat}: {len(txt_files)} files")
        print(f"{'='*60}")

        for filepath in txt_files:
            try:
                if cat == "Textbooks":
                    n_chapters, n_chunks = process_textbook(filepath, out_dir)
                    stats["chapters"] += n_chapters
                    stats["chunks"] += n_chunks
                    stats["docs"] += 1
                    print(f"  [ok] {filepath.name}: {n_chapters} chapters, {n_chunks} chunks")
                else:
                    n_chunks = process_single_file(filepath, out_dir)
                    if n_chunks > 0:
                        stats["docs"] += 1
                        stats["chunks"] += n_chunks
                        print(f"  [ok] {filepath.name}: {n_chunks} chunks")
                    else:
                        stats["skipped"] += 1
                        print(f"  [skip] {filepath.name}: too short")
            except Exception as e:
                print(f"  [err] {filepath.name}: {e}")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Documents processed: {stats['docs']}")
    print(f"  Textbook chapters: {stats['chapters']}")
    print(f"  Total chunks written: {stats['chunks']}")
    print(f"  Skipped (too short): {stats['skipped']}")


if __name__ == "__main__":
    main()
