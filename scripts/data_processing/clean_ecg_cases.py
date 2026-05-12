import re
from pathlib import Path
import json
from typing import List, Tuple

def clean_ecg_case_content(content: str) -> str:
    original_length = len(content)

    page_patterns = [
        r'--- Страница \d+ ---\s*\n\d{2}/\d{2}/\d{4}, \d{2}:\d{2}.*?- Dr\. Smith’s ECG Blog\s*\n',
        r'--- Страница \d+ ---\s*\n',
        r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2}.*?- Dr\. Smith’s ECG Blog',
        r'https://drsmithsecgblog\.com/.*?/\d+/\d+\s*\n',
    ]

    for pattern in page_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    navigation_patterns = [
        r'Dr\. Smith\'s ECG Blog\s*\nInstructive ECGs in Emergency Medicine Clinical Content\s*\n'
        r'Associate Editors:.*?Home.*?\n',
        r'Home.*?\n(?:.*?\n){0,3}',
    ]

    for pattern in navigation_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

    if 'Write a Comment' in content:
        comment_start = content.find('Write a Comment')
        if comment_start > 0:
            before_comment = content[:comment_start]
            medical_keywords = ['ECG', 'chest pain', 'patient', 'diagnosis', 'treatment', 'STEMI', 'OMI']

            if any(keyword.lower() in before_comment.lower() for keyword in medical_keywords):
                content = before_comment.strip()

    about_patterns = [
        r'ABOUT.*?FOLLOW US ON X \(TWITTER\).*?(?=\n\n|\Z)',
        r'FOLLOW US ON X \(TWITTER\).*?FEATURED POSTS.*?(?=\n\n|\Z)',
        r'FEATURED POSTS.*?BLOG ARCHIVE.*?(?=\n\n|\Z)',
        r'BLOG ARCHIVE.*?Select Month.*?(?=\n\n|\Z)',
        r'LABELS.*?Read Next.*?(?=\n\n|\Z)',
        r'Read Next.*?Never Miss a Beat.*?(?=\n\n|\Z)',
        r'Never Miss a Beat.*?Expert ECG Interpretation.*?(?=\n\n|\Z)',
        r'© \d{4} — Dr\. Smith\'s ECG Blog\..*?(?=\n\n|\Z)',
        r'This work is licensed under.*?International License\.',
        r'Follow @\w+\s*',
    ]

    for pattern in about_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

    trash_lines = [
        'Trusted insights, no spam—only ECG brilliance.',
        'Expert ECG Interpretation and Emergency Cardiology Education',
        'Get the latest expert ECG cases, clinical pearls, and interpretation tips',
        'Email Address Subscribe',
        'Dr. Smith\'s Google Scholar Profile',
        'Dr. Smith Articles on PubMed',
        'FACULTY PHYSICIAN',
        r'Written by .*? on.*?\d{4}',
        r'This was written by .*?\..*?\n',
        r'This was sent by .*?\..*?\n',
    ]

    for line_pattern in trash_lines:
        content = re.sub(line_pattern + r'.*?\n', '', content, flags=re.IGNORECASE)

    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        if line.count('"') >= 4:
            medical_indicators = ['ECG', 'pain', 'patient', 'heart', 'chest', 'diagnos', 'treat']
            if not any(indicator.lower() in line.lower() for indicator in medical_indicators):
                continue
        cleaned_lines.append(line)

    content = '\n'.join(cleaned_lines)

    tag_pattern = r'\"[^\"]+\"\(?\d*\)?\s*'
    content = re.sub(tag_pattern, '', content)

    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r'[ \t]{2,}', ' ', content)
    content = content.strip()

    medical_keywords = ['ECG', 'patient', 'chest', 'pain', 'heart', 'diagnosis',
                       'treatment', 'history', 'symptoms', 'findings', 'case']

    has_medical_content = any(keyword.lower() in content.lower() for keyword in medical_keywords)

    if not has_medical_content or len(content) < 100:
        print(f"  Warning: possible removal of medical content")
        print(f"     Length: {len(content)} symbols")
        return None

    cleaned_length = len(content)
    print(f"  Cleaned: {original_length} → {cleaned_length} chars ({cleaned_length/original_length*100:.1f}%)")

    return content

def process_ecg_case_file(file_path: Path) -> Tuple[bool, int, int]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_content = f.read()

        if '=== СОДЕРЖАНИЕ ===' not in full_content:
            return False, 0, 0

        metadata_part, case_content = full_content.split('=== СОДЕРЖАНИЕ ===', 1)

        is_ecg_case = any(keyword in full_content for keyword in
                         ['Dr. Smith', 'ECG Blog', 'chest pain', 'ECG'])

        if not is_ecg_case:
            return False, 0, 0

        print(f"Processing: {file_path.name}")

        cleaned_content = clean_ecg_case_content(case_content)

        if cleaned_content is None:
            print(f"  File not processed: insufficient medical content")
            return False, len(case_content), 0

        new_content = metadata_part + '=== СОДЕРЖАНИЕ ===\n' + cleaned_content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True, len(case_content), len(cleaned_content)

    except Exception as e:
        print(f"  Error processing {file_path.name}: {str(e)}")
        return False, 0, 0

def analyze_case_file(file_path: Path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if '=== СОДЕРЖАНИЕ ===' in content:
        _, case_content = content.split('=== СОДЕРЖАНИЕ ===', 1)

        trash_patterns = {
            'страницы': r'--- Страница \d+ ---',
            'даты': r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2}',
            'url': r'https://drsmithsecgblog\.com',
            'навигация': r'Dr\. Smith\'s ECG Blog',
            'теги': r'LABELS',
            'реклама': r'Never Miss a Beat',
            'комментарии': r'Write a Comment',
        }

        trash_counts = {}
        for name, pattern in trash_patterns.items():
            matches = len(re.findall(pattern, case_content, re.IGNORECASE))
            trash_counts[name] = matches

        medical_keywords = ['ECG', 'chest pain', 'patient', 'diagnosis', 'STEMI', 'OMI', 'angina']
        medical_count = sum(1 for kw in medical_keywords if kw.lower() in case_content.lower())

        return {
            'file': file_path.name,
            'total_length': len(case_content),
            'trash_patterns': trash_counts,
            'medical_keywords': medical_count,
            'has_ecg_blog': 'Dr. Smith' in case_content or 'ECG Blog' in case_content,
        }
    return {}

def main():
    cases_path = Path("data/processed/cardiology/Cases")

    if not cases_path.exists():
        print(f"Cases folder not found: {cases_path}")
        return

    print("Analyzing case files...")
    txt_files = list(cases_path.glob("*.txt"))
    print(f"Files found: {len(txt_files)}")

    ecg_cases = []
    other_cases = []

    for file_path in txt_files[:10]:
        analysis = analyze_case_file(file_path)
        if analysis:
            if analysis['has_ecg_blog']:
                ecg_cases.append(analysis)
            else:
                other_cases.append(analysis)

    print(f"\nResults of analysis (first 10 files):")
    print(f"Dr. Smith's ECG Blog cases: {len(ecg_cases)}")
    print(f"Other cases: {len(other_cases)}")

    if ecg_cases:
        print("\nSample of trash in ECG cases:")
        for pattern, count in ecg_cases[0]['trash_patterns'].items():
            if count > 0:
                print(f"  - {pattern}: {count}")

    print(f"\n{'='*60}")
    print("ОЧИСТКА КЕЙСОВ DR. SMITH'S ECG BLOG")
    print("="*60)
    print("What will be deleted:")
    print("  • Page headers (--- Page X ---")
    print("  • Dates and URL")
    print("  • Blog navigation")
    print("  • ABOUT block and ads")
    print("  • Tags (LABELS)")
    print("  • Subscription form")
    print("  • Copyright footer")

    response = input("\nContinue cleaning? (y/n): ").lower()
    if response != 'y':
        print("Cleaning cancelled.")
        return

    print(f"\n Processing {len(txt_files)} files...")

    total_original = 0
    total_cleaned = 0
    processed_count = 0

    for file_path in txt_files:
        success, orig_len, cleaned_len = process_ecg_case_file(file_path)
        if success:
            total_original += orig_len
            total_cleaned += cleaned_len
            processed_count += 1

    print(f"\n{'='*60}")
    print("RESULTS OF CLEANING")
    print("="*60)
    print(f"Processed files: {processed_count}/{len(txt_files)}")
    print(f"Total size:")
    print(f"  Before cleaning: {total_original:,} symbols")
    print(f"  After cleaning: {total_cleaned:,} symbols")
    if total_original > 0:
        reduction = (total_original - total_cleaned) / total_original * 100
        print(f"  Удалено: {reduction:.1f}% мусора")

    report = {
        'total_files': len(txt_files),
        'processed_files': processed_count,
        'total_original_chars': total_original,
        'total_cleaned_chars': total_cleaned,
        'reduction_percent': reduction if total_original > 0 else 0,
        'timestamp': __import__('datetime').datetime.now().isoformat(),
    }

    report_path = cases_path / "cleaning_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved: {report_path}")

    if processed_count > 0:
        print(f"\nExample of cleaned file:")
        sample_file = txt_files[0]
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if '=== СОДЕРЖАНИЕ ===' in content:
            _, sample_content = content.split('=== СОДЕРЖАНИЕ ===', 1)
            print("\n" + "="*40)
            print("ПЕРВЫЕ 500 СИМВОЛОВ:")
            print("="*40)
            print(sample_content[:500] + "...")
            print("="*40)

def quick_clean_single_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Файл не найден: {file_path}")
        return

    success, orig_len, cleaned_len = process_ecg_case_file(path)
    if success:
        print(f"\n File cleaned: {path.name}")
        print(f"   Было: {orig_len:,} символов")
        print(f"   Стало: {cleaned_len:,} символов")
        print(f"   Удалено: {(orig_len - cleaned_len)/orig_len*100:.1f}%")
    else:
        print(f"\n Не удалось очистить файл")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        quick_clean_single_file(sys.argv[1])
    else:
        main()
