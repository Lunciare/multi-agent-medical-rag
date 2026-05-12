import asyncio
import io
import os
import random
import re
import time
from urllib.parse import urljoin, urlparse

import fitz
import requests
from bs4 import BeautifulSoup, Comment


TARGET_BOOKS = {
    "Endotext": "https://www.ncbi.nlm.nih.gov/books/NBK278943/",
    "NCBI_Book_568737": "https://www.ncbi.nlm.nih.gov/books/NBK568737/",
    "Anatomy_OpenStax": "https://openstax.org/books/anatomy-and-physiology-2e/pages/",
    "Diabetes_America": "https://www.niddk.nih.gov/about-niddk/strategic-plans-reports/diabetes-in-america-3rd-edition",
    "NCBI_Book_430685": "https://www.ncbi.nlm.nih.gov/books/NBK430685/",
    "ThyroidManager": "https://www.thyroidmanager.org/",
}

OUTPUT_DIR = os.path.join("data", "raw", "Endocrinology", "Textbooks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "MedicalRAGCrawler/1.0 (Academic Research; "
        "Multi-Agent-NN-Medicine project; contact: research@example.com)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

CHAPTER_SEPARATOR = (
    "\n\n"
    + "═" * 60
    + "\nCHAPTER: {title}\nSOURCE:  {url}\n"
    + "═" * 60
    + "\n\n"
)


def polite_sleep(min_sec=2.5, max_sec=5.0):
    time.sleep(random.uniform(min_sec, max_sec))


def fetch_html(url: str, session: requests.Session | None = None) -> str | None:
    s = session or requests.Session()
    try:
        resp = s.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  [!] Failed to fetch {url}: {e}")
        return None


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    for tag_name in ["script", "style", "noscript", "iframe", "svg"]:
        for el in soup.find_all(tag_name):
            el.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    boilerplate_selectors = [
        "nav", "header", "footer",
        ".pmc-sidebar", ".rprt_nav", ".col-3",
        ".copyright-statement", ".disclaimer",
        "#ncbi-header", "#ncbi-footer", "#ncbitoolbar",
        ".icnblk_cntnt",  # NCBI toolbar buttons
        ".pmc-wm",        # NCBI watermark
        "[role='navigation']",
        "[role='banner']",
        "[role='contentinfo']",
        ".nav-inner", ".navbar",
        ".skip-to-main", ".usa-banner",
        "#global-header", "#global-footer",
        ".share-links", ".related-links",
    ]
    for selector in boilerplate_selectors:
        for el in soup.select(selector):
            el.decompose()

    return soup


def soup_to_clean_text(soup: BeautifulSoup) -> str:
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class NCBIExtractor:
    NCBI_BASE = "https://www.ncbi.nlm.nih.gov"

    def __init__(self, session: requests.Session):
        self.session = session

    def get_chapter_links(self, toc_url: str) -> list[tuple[str, str]]:
        html = fetch_html(toc_url, self.session)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        chapters = []
        seen = set()

        for a_tag in soup.select("a[href]"):
            href = a_tag["href"].split("#")[0]
            if not href:
                continue

            full_url = urljoin(toc_url, href)

            is_nbk = re.search(r"/books/NBK\d+/?$", full_url)
            is_named = re.search(r"/books/n/[^/]+/[^/]+/?$", full_url)

            if not (is_nbk or is_named):
                continue

            title = a_tag.get_text(strip=True) or "Untitled"
            if title.lower() in ("help", "next >", "< prev", "expand all", "collapse all", "views", "cite", "share", "related information", "pubmed"):
                continue

            if full_url.rstrip("/") == toc_url.rstrip("/"):
                continue

            if full_url not in seen:
                seen.add(full_url)
                title = a_tag.get_text(strip=True) or "Untitled"
                chapters.append((title, full_url))

        return chapters

    def extract_chapter_content(self, url: str) -> str | None:
        html = fetch_html(url, self.session)
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        content_el = (
            soup.select_one("div.body-content")
            or soup.select_one("div#mc")
            or soup.select_one("div.jig-ncbiinpagenav")
            or soup.select_one("article")
            or soup.select_one("main")
        )

        if not content_el:
            return None

        clean_soup(content_el)

        for el in content_el.select(".ref-list, .ref-cit, .rprt_nav, .link-list"):
            el.decompose()

        text = soup_to_clean_text(content_el)

        if len(text) < 200:
            return None

        return text

    def extract_book(self, book_name: str, toc_url: str) -> str | None:
        print(f"\n{'━' * 60}")
        print(f"[{book_name}] Starting NCBI Bookshelf extraction")
        print(f"  TOC: {toc_url}")
        print(f"{'━' * 60}")

        chapters = self.get_chapter_links(toc_url)
        if not chapters:
            print(f"  [!] No chapter links found at {toc_url}")
            return self.extract_single_article(book_name, toc_url)

        print(f"  Found {len(chapters)} chapter links")
        all_text = []

        for i, (title, url) in enumerate(chapters, 1):
            print(f"  → [{i}/{len(chapters)}] {title[:80]}...")
            polite_sleep(3.0, 6.0)

            text = self.extract_chapter_content(url)
            if text:
                separator = CHAPTER_SEPARATOR.format(title=title, url=url)
                all_text.append(separator + text)
                print(f"    ✓ Extracted {len(text):,} characters")
            else:
                print(f"    ✗ No content extracted")

        if not all_text:
            return None

        return "\n".join(all_text)

    def extract_single_article(self, book_name: str, url: str) -> str | None:
        print(f"\n[{book_name}] Extracting single article: {url}")
        text = self.extract_chapter_content(url)
        if text:
            title = book_name.replace("_", " ")
            separator = CHAPTER_SEPARATOR.format(title=title, url=url)
            return separator + text
        return None


class OpenStaxExtractor:
    TOC_URL = "https://openstax.org/books/anatomy-and-physiology-2e/pages/1-introduction"

    async def get_all_page_urls(self, browser) -> list[tuple[str, str]]:
        page = await browser.new_page()
        try:
            await page.goto(self.TOC_URL, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(3)  # React render

            links = await page.evaluate("""
                () => {
                    const results = [];
                    const anchors = document.querySelectorAll(
                        'nav a[href*="/pages/"], ol.os-toc a[href*="/pages/"]'
                    );
                    for (const a of anchors) {
                        const href = a.getAttribute('href');
                        if (href && href.includes('/pages/') && !href.includes('table-of-contents')) {
                            results.push({
                                title: a.textContent.trim(),
                                url: new URL(href, window.location.origin).href
                            });
                        }
                    }
                    const seen = new Set();
                    return results.filter(r => {
                        if (seen.has(r.url)) return false;
                        seen.add(r.url);
                        return true;
                    });
                }
            """)
            return [(l["title"], l["url"]) for l in links]
        finally:
            await page.close()

    async def extract_page_content(self, browser, url: str) -> str | None:
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(2)

            content = await page.evaluate("""
                () => {
                    const el = document.querySelector('[data-type="page"]')
                              || document.querySelector('.page-inner-content')
                              || document.querySelector('#main-content')
                              || document.querySelector('main');
                    if (!el) return null;

                    const removeSelectors = [
                        'nav', 'header', 'footer',
                        '.os-teacher-note', '.os-eoc', '.os-eos',
                        '[data-type="metadata"]',
                        '.os-toolbar', '.prev-next-nav',
                        '.citation-section', '.os-reference-section',
                        '.os-eob', '.os-solutions-container',
                    ];
                    for (const sel of removeSelectors) {
                        el.querySelectorAll(sel).forEach(e => e.remove());
                    }
                    return el.innerText;
                }
            """)
            if content and len(content.strip()) > 100:
                return content.strip()
            return None
        except Exception as e:
            print(f"    [!] Error extracting {url}: {e}")
            return None
        finally:
            await page.close()

    async def extract_book(self, book_name: str, browser) -> str | None:
        print(f"\n{'━' * 60}")
        print(f"[{book_name}] Starting OpenStax extraction (Playwright)")
        print(f"{'━' * 60}")

        pages = await self.get_all_page_urls(browser)
        if not pages:
            print(f"  [!] No page links discovered")
            return None

        print(f"  Found {len(pages)} pages")
        all_text = []

        for i, (title, url) in enumerate(pages, 1):
            print(f"  → [{i}/{len(pages)}] {title[:80]}...")
            await asyncio.sleep(random.uniform(1.5, 3.0))

            text = await self.extract_page_content(browser, url)
            if text:
                separator = CHAPTER_SEPARATOR.format(title=title, url=url)
                all_text.append(separator + text)
                print(f"    ✓ Extracted {len(text):,} characters")
            else:
                print(f"    ✗ No content extracted")

        return "\n".join(all_text) if all_text else None


class NIDDKExtractor:
    def __init__(self, session: requests.Session):
        self.session = session

    def get_pdf_links(self, toc_url: str) -> list[tuple[str, str]]:
        html = fetch_html(toc_url, self.session)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        pdfs = []
        seen = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if ".pdf" not in href.lower():
                continue
            if "_Figures" in href or ".zip" in href.lower():
                continue

            full_url = urljoin(toc_url, href)
            if full_url in seen:
                continue
            seen.add(full_url)

            title = a_tag.get_text(strip=True)
            if not title or len(title) < 5:
                parent = a_tag.find_parent("li")
                if parent:
                    title = parent.get_text(strip=True).split("(PDF")[0].strip()
                else:
                    title = os.path.basename(href).replace(".pdf", "")

            pdfs.append((title, full_url))

        return pdfs

    def extract_pdf_text(self, url: str) -> str | None:
        try:
            resp = self.session.get(url, headers=HEADERS, timeout=120)
            resp.raise_for_status()

            doc = fitz.open(stream=resp.content, filetype="pdf")
            text_parts = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(text.strip())

            doc.close()
            full_text = "\n\n".join(text_parts)
            return full_text if len(full_text) > 100 else None

        except Exception as e:
            print(f"    [!] PDF extraction failed for {url}: {e}")
            return None

    def extract_book(self, book_name: str, toc_url: str) -> str | None:
        print(f"\n{'━' * 60}")
        print(f"[{book_name}] Starting NIDDK PDF extraction")
        print(f"  TOC: {toc_url}")
        print(f"{'━' * 60}")

        pdfs = self.get_pdf_links(toc_url)
        if not pdfs:
            print(f"  [!] No PDF links found")
            return None

        print(f"  Found {len(pdfs)} PDF chapters")
        all_text = []

        for i, (title, url) in enumerate(pdfs, 1):
            print(f"  → [{i}/{len(pdfs)}] {title[:80]}...")
            polite_sleep(2.0, 4.0)

            text = self.extract_pdf_text(url)
            if text:
                separator = CHAPTER_SEPARATOR.format(title=title, url=url)
                all_text.append(separator + text)
                print(f"    ✓ Extracted {len(text):,} characters")
            else:
                print(f"    ✗ No content extracted")

        return "\n".join(all_text) if all_text else None


class ThyroidManagerExtractor:
    def __init__(self, session: requests.Session):
        self.session = session

    def get_chapter_links(self, home_url: str) -> list[tuple[str, str]]:
        html = fetch_html(home_url, self.session)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        chapters = []
        seen = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/chapter/" not in href:
                continue

            full_url = href if href.startswith("http") else urljoin(home_url, href)
            full_url = full_url.rstrip("/") + "/"

            if full_url in seen:
                continue
            seen.add(full_url)

            title = a_tag.get_text(strip=True) or "Untitled"
            chapters.append((title, full_url))

        return chapters

    def extract_chapter_content(self, url: str) -> str | None:
        html = fetch_html(url, self.session)
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        content_el = (
            soup.select_one(".entry-content")
            or soup.select_one("article .post-content")
            or soup.select_one("article")
            or soup.select_one("#content")
            or soup.select_one("main")
        )

        if not content_el:
            return None

        clean_soup(content_el)

        for el in content_el.select(
            ".sharedaddy, .wpcnt, .post-navigation, "
            ".comment-section, .comments, #respond, "
            ".post-meta, .entry-meta, .post-tags"
        ):
            el.decompose()

        text = soup_to_clean_text(content_el)
        return text if len(text) > 200 else None

    def extract_book(self, book_name: str, home_url: str) -> str | None:
        print(f"\n{'━' * 60}")
        print(f"[{book_name}] Starting ThyroidManager extraction")
        print(f"  Home: {home_url}")
        print(f"{'━' * 60}")

        chapters = self.get_chapter_links(home_url)
        if not chapters:
            print(f"  [!] No chapter links found")
            return None

        print(f"  Found {len(chapters)} chapters")
        all_text = []

        for i, (title, url) in enumerate(chapters, 1):
            print(f"  → [{i}/{len(chapters)}] {title[:80]}...")
            polite_sleep(2.0, 4.0)

            text = self.extract_chapter_content(url)
            if text:
                separator = CHAPTER_SEPARATOR.format(title=title, url=url)
                all_text.append(separator + text)
                print(f"    ✓ Extracted {len(text):,} characters")
            else:
                print(f"    ✗ No content (may require login/registration)")

        return "\n".join(all_text) if all_text else None


def save_book(book_name: str, content: str):
    filename = f"{book_name}_Complete.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{'═' * 60}\n")
        f.write(f"  {book_name.replace('_', ' ').upper()}\n")
        f.write(f"  Extracted for RAG pipeline\n")
        f.write(f"{'═' * 60}\n\n")
        f.write(content)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n  ✓ Saved: {filepath}  ({size_mb:.2f} MB)")


def classify_source(book_name: str, url: str) -> str:
    if "ncbi.nlm.nih.gov/books" in url:
        return "ncbi"
    elif "openstax.org" in url:
        return "openstax"
    elif "niddk.nih.gov" in url:
        return "niddk"
    elif "thyroidmanager.org" in url:
        return "thyroidmanager"
    else:
        return "ncbi"


async def main():
    print("=" * 60)
    print("  Intelligent Medical RAG Crawler v2.0")
    print("  Output: data/raw/Endocrinology/Textbooks/")
    print("=" * 60)

    session = requests.Session()
    session.headers.update(HEADERS)

    ncbi = NCBIExtractor(session)
    niddk = NIDDKExtractor(session)
    thyroid = ThyroidManagerExtractor(session)
    openstax = OpenStaxExtractor()

    results = {}

    for book_name, url in TARGET_BOOKS.items():
        source_type = classify_source(book_name, url)

        if source_type == "openstax":
            continue

        content = None
        if source_type == "ncbi":
            content = ncbi.extract_book(book_name, url)
        elif source_type == "niddk":
            content = niddk.extract_book(book_name, url)
        elif source_type == "thyroidmanager":
            content = thyroid.extract_book(book_name, url)

        if content:
            save_book(book_name, content)
            results[book_name] = "✓ Success"
        else:
            results[book_name] = "✗ Failed"
            print(f"\n  [!] Failed to extract content for {book_name}")

    has_openstax = any(
        classify_source(name, url) == "openstax"
        for name, url in TARGET_BOOKS.items()
    )

    if has_openstax:
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                for book_name, url in TARGET_BOOKS.items():
                    if classify_source(book_name, url) != "openstax":
                        continue
                    content = await openstax.extract_book(book_name, browser)
                    if content:
                        save_book(book_name, content)
                        results[book_name] = "✓ Success"
                    else:
                        results[book_name] = "✗ Failed"
                        print(f"\n  [!] Failed to extract content for {book_name}")
                await browser.close()

        except ImportError:
            print("\n  [!] Playwright not installed. Skipping OpenStax.")
            print("      Install with: pip install playwright && playwright install chromium")
            for book_name, url in TARGET_BOOKS.items():
                if classify_source(book_name, url) == "openstax":
                    results[book_name] = "✗ Skipped (no Playwright)"

    print(f"\n\n{'═' * 60}")
    print("  EXTRACTION SUMMARY")
    print(f"{'═' * 60}")
    for book_name, status in results.items():
        print(f"  {status}  {book_name}")
    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
