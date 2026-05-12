import os
import shutil
import fitz

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def process_files(raw_dir, processed_dir):
    ensure_dir(processed_dir)

    success_count = 0
    fail_count = 0
    skipped_count = 0
    failed_files = []

    print(f"Starting processing from '{raw_dir}' to '{processed_dir}'...\n")

    for root, dirs, files in os.walk(raw_dir):
        rel_path = os.path.relpath(root, raw_dir)
        target_dir = os.path.join(processed_dir, rel_path)

        ensure_dir(target_dir)

        for filename in files:
            source_file = os.path.join(root, filename)

            if filename.startswith('.'):
                continue

            try:
                if filename.lower().endswith('.txt'):
                    target_file = os.path.join(target_dir, filename)
                    print(f"Copying TXT: {filename}")
                    shutil.copy2(source_file, target_file)
                    success_count += 1

                elif filename.lower().endswith('.pdf'):
                    base_name = os.path.splitext(filename)[0]
                    target_file = os.path.join(target_dir, f"{base_name}.txt")
                    print(f"Extracting PDF: {filename} -> {base_name}.txt")

                    text_content = []
                    with fitz.open(source_file) as doc:
                        for page in doc:
                            text_content.append(page.get_text("text"))

                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write("\n".join(text_content))

                    success_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"  [Error] Failed to process {filename}: {e}")
                failed_files.append((source_file, str(e)))
                fail_count += 1

    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process:      {fail_count} files")
    print(f"Skipped (unsupported):  {skipped_count} files")

    if failed_files:
        print("\nFailed Files Details:")
        for file, err in failed_files:
            print(f" - {os.path.basename(file)}: {err}")
    print("="*50)

if __name__ == "__main__":
    RAW_DIR = "data/raw/Endocrinology"
    PROCESSED_DIR = "data/processed/endocrinology"

    if not os.path.exists(RAW_DIR):
        print(f"Error: Source directory '{RAW_DIR}' does not exist.")
    else:
        process_files(RAW_DIR, PROCESSED_DIR)
