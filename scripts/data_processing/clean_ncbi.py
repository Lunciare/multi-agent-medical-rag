import os
import re

def clean_text(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Internal file not found at {input_file}")
        return

    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    chapters = re.split(r'═{40,}\nCHAPTER:.*?\nSOURCE:.*?\n═{40,}\n\n', content)
    header = chapters[0] if chapters else ""
    chapter_bodies = chapters[1:] if len(chapters) > 1 else []

    parts = re.split(r'(═{40,}\nCHAPTER:.*?\nSOURCE:.*?\n═{40,}\n\n)', content)

    cleaned_parts = []

    author_regex = re.compile(
        r'\b(md|phd|do|mbbs|bsn|msn|aprn|fnp|dnp|rn|np|dpm|pharmd|facs|facp|faap|msc|mph|dmd|dds|od|bds|mba)\\b|'
        r'\b(professor|assistant professor|associate professor|resident|practicing physician|nurse|director|fellow|student|specialist|clinical|adjunct)\b|'
        r'\b(university|hospital|clinic|medical center|college|school of medicine|health|department)\b',
        re.IGNORECASE
    )

    boilerplate_phrases = [
        "Review Questions",
        "Access free multiple choice questions on this topic.",
        "Click here for a simplified version",
        "Comment on this article.",
        "Go to:",
        "Disclosure:"
    ]

    print("Cleaning chapters...")
    for part in parts:
        if part.startswith('═' * 40):
            cleaned_parts.append(part)
            continue

        if not part.strip() and not cleaned_parts:
            cleaned_parts.append(part)
            continue

        lines = part.split('\n')
        cleaned_lines = []

        in_references = False

        for idx, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                cleaned_lines.append(line)
                continue

            if re.match(r'^(References|Bibliography)$', stripped, re.IGNORECASE):
                in_references = True
                continue

            if in_references:
                continue

            if any(bp.lower() in stripped.lower() for bp in boilerplate_phrases):
                continue

            if re.match(r'^\[\s*(PubMed|PMC free article)?\s*:?\s*\d+\s*\]$', stripped):
                continue
            if re.match(r'^\[\s*\d+\s*\]$', stripped):
                continue

            if len(stripped) < 300 and ',' in stripped:
                segments = [s.strip() for s in stripped.split(',')]
                match_count = sum(1 for s in segments if author_regex.search(s))

                if match_count >= 1 and len(segments) >= 2:
                    continue

            cleaned_lines.append(line)

        cleaned_chapter = '\n'.join(cleaned_lines)
        cleaned_chapter = re.sub(r'\n{3,}', '\n\n', cleaned_chapter)

        cleaned_parts.append(cleaned_chapter)

    final_content = "".join(cleaned_parts)

    print(f"Writing sanitized output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_content)

    orig_size = os.path.getsize(input_file) / (1024*1024)
    new_size = os.path.getsize(output_file) / (1024*1024)
    print(f"Done! Reduced size from {orig_size:.2f} MB to {new_size:.2f} MB")

if __name__ == "__main__":
    input_path = "data/raw/Endocrinology/Textbooks/NCBI_Book_430685_Complete.txt"
    output_path = "data/raw/Endocrinology/Textbooks/NCBI_Book_430685_Cleaned.txt"
    clean_text(input_path, output_path)
