import re
with open("scripts/data_processing/rag_crawler.py", "r") as f:
    code = f.read()

old_block = """            if not (is_nbk or is_named):
                continue

            # Don't re-visit the TOC itself
            if full_url.rstrip("/") == toc_url.rstrip("/"):
                continue"""
new_block = """            if not (is_nbk or is_named):
                continue

            title = a_tag.get_text(strip=True) or "Untitled"
            if title.lower() in ("help", "next >", "< prev", "expand all", "collapse all", "views", "cite", "share", "related information", "pubmed"):
                continue

            if full_url.rstrip("/") == toc_url.rstrip("/"):
                continue"""

code = code.replace(old_block, new_block)
with open("scripts/data_processing/rag_crawler.py", "w") as f:
    f.write(code)
