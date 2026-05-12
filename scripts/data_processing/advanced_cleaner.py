import os
import re

PROCESSED_DIR = "data/processed/endocrinology"

REGEX_URLS = re.compile(r'https?://[^\s]+')
REGEX_EMAILS = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
REGEX_DOIS = re.compile(r'\b10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+\b')

REGEX_BRACKET_CITATIONS = re.compile(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]')
REGEX_AUTHOR_YEAR = re.compile(r'\([A-Z][a-zA-Z-]+\s+(?:et al\.,?\s+)?\d{4}[a-z]?(?:\s*,\s*[A-Z][a-zA-Z-]+\s+(?:et al\.,?\s+)?\d{4}[a-z]?)*\)')

REGEX_PATIENT_INITIALS = re.compile(r'\b(?:Mr\.|Mrs\.|Ms\.|Patient)\s+(?:[A-Z]\.?[A-Z]?\.?)\b')
REGEX_DATES = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')

METADATA_KEYWORDS = re.compile(
    r'\b(md|phd|do|mbbs|bsn|rn|msn|ba|bsc|msc|pharmd|facs|facp|faap|mph|dmd|dds|od|bds|mba)\b|'
    r'\b(professor|assistant professor|associate professor|resident|physician|nurse|director|fellow|student|specialist|clinical|adjunct)\b|'
    r'\b(university|hospital|clinic|medical center|college|health|department|institute|school of medicine)\b|'
    r'published online|received:|accepted:|copyright|all rights reserved|objective:|methods:|results:|conclusion:',
    re.IGNORECASE
)

BOILERPLATE_EXACT = [
    "click here", "review questions", "comment on this article",
    "go to:", "disclosure:", "access free multiple choice questions"
]

def clean_text(text):
    chapters = re.split(r'(═{40,}\nCHAPTER:.*?\nSOURCE:.*?\n═{40,}\n\n)', text)
    if len(chapters) == 1:
        text = remove_references(text)
    else:
        cleaned_parts = []
        for part in chapters:
            if part.startswith('═' * 40):
                cleaned_parts.append(part)
            else:
                cleaned_parts.append(remove_references(part))
        text = "".join(cleaned_parts)

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append(line)
            continue

        if any(bp in stripped.lower() for bp in BOILERPLATE_EXACT):
            continue

        if re.match(r'^\d+$', stripped) or re.match(r'^page \d+ of \d+$', stripped.lower()):
            continue

        if len(stripped) < 300 and ',' in stripped:
            segments = [s.strip() for s in stripped.split(',')]
            match_count = sum(1 for s in segments if METADATA_KEYWORDS.search(s))
            if match_count >= 1 and len(segments) >= 2:
                continue

        line = REGEX_URLS.sub('', line)
        line = REGEX_DOIS.sub('', line)
        line = REGEX_EMAILS.sub('', line)

        line = REGEX_BRACKET_CITATIONS.sub('', line)
        line = REGEX_AUTHOR_YEAR.sub('', line)

        line = REGEX_PATIENT_INITIALS.sub('[PATIENT]', line)
        line = REGEX_DATES.sub('[DATE]', line)

        line = re.sub(r'\s+', ' ', line).strip()

        if line:
            cleaned_lines.append(line)

    cleaned_doc = '\n'.join(cleaned_lines)
    cleaned_doc = re.sub(r'\n{3,}', '\n\n', cleaned_doc)
    return cleaned_doc

def remove_references(text):
    lines = text.split('\n')
    out = []
    for line in lines:
        stripped = line.strip().lower()
        if re.match(r'^(references|bibliography|literature cited|works cited)[.:]*$', stripped):
            break
        out.append(line)
    return '\n'.join(out)

def process_directory(directory):
    total_files = 0
    total_original_chars = 0
    total_cleaned_chars = 0

    print(f"{'File Name':<50} | {'Original Size':<15} | {'Cleaned Size':<15} | {'Reduction'}")
    print("-" * 105)

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.txt') and not filename.lower().endswith('_cleaned.txt'):
                filepath = os.path.join(root, filename)

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()

                    original_len = len(text)
                    if original_len == 0:
                        continue

                    cleaned_text = clean_text(text)
                    cleaned_len = len(cleaned_text)

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)

                    total_files += 1
                    total_original_chars += original_len
                    total_cleaned_chars += cleaned_len

                    reduction = ((original_len - cleaned_len) / original_len) * 100

                    display_name = (filename[:47] + '...') if len(filename) > 50 else filename
                    print(f"{display_name:<50} | {original_len:<15,} | {cleaned_len:<15,} | {reduction:.1f}%")

                except Exception as e:
                    print(f"[Error] Processing {filename}: {e}")

    print("-" * 105)
    print(f"Total Files Cleaned: {total_files}")
    if total_original_chars > 0:
        total_red = ((total_original_chars - total_cleaned_chars) / total_original_chars) * 100
        print(f"Total Original Characters: {total_original_chars:,}")
        print(f"Total Cleaned Characters:  {total_cleaned_chars:,}")
        print(f"Overall Garbage Removed:   {total_red:.1f}%")

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DIR):
        print(f"Directory {PROCESSED_DIR} does not exist.")
    else:
        print(f"Starting advanced cleaning in {PROCESSED_DIR}...")
        process_directory(PROCESSED_DIR)
