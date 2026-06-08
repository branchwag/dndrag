"""
Removes YouTube references from every PDF in ./docs:
  - Link annotations pointing to YouTube URLs (deleted)
  - Visible text tokens containing "youtube" / "youtu.be" (redacted to white)
Overwrites each file in place after changes.
"""

import os
import sys
import glob
import fitz  # PyMuPDF

DOCS_DIR = "docs"
YOUTUBE_KEYWORDS = ("youtube.com", "youtu.be", "youtube")


def is_youtube(s: str) -> bool:
    low = s.lower()
    return any(k in low for k in YOUTUBE_KEYWORDS)


def clean_pdf(path: str) -> bool:
    doc = fitz.open(path)
    changed = False

    for page in doc:
        # --- 1. Remove link annotations pointing to YouTube ---
        links = page.get_links()
        for link in links:
            uri = link.get("uri", "")
            if is_youtube(uri):
                page.delete_link(link)
                changed = True
                print(f"  removed link annotation: {uri[:80]}")

        # --- 2. Redact any visible text tokens containing YouTube ---
        words = page.get_text("words")
        for x0, y0, x1, y1, word, *_ in words:
            if is_youtube(word):
                page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(1, 1, 1))
                changed = True
                print(f"  redacted text token: {word!r}")

        if changed:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    if changed:
        tmp = path + ".tmp"
        doc.save(tmp, incremental=False, encryption=fitz.PDF_ENCRYPT_NONE)
        doc.close()
        os.replace(tmp, path)
        print(f"  saved: {path}")
    else:
        doc.close()
        print(f"  no YouTube references found: {path}")

    return changed


def main():
    pdfs = glob.glob(f"{DOCS_DIR}/*.pdf")
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}/")
        sys.exit(0)

    total_changed = 0
    for pdf in sorted(pdfs):
        if clean_pdf(pdf):
            total_changed += 1

    print(f"\nDone. {total_changed}/{len(pdfs)} file(s) modified.")
    if total_changed:
        print("Re-run 'make ingest ARGS=\"--fresh\"' to re-index the cleaned documents.")


if __name__ == "__main__":
    main()
