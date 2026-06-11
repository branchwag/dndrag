"""
Cleans every PDF in ./docs:
  1. Removes link annotations pointing to filtered URLs
  2. Redacts visible text tokens containing FILTER_KEYWORDS (blanked to white)
  3. Strips all embedded images (reduces file size and extraction noise)

Run without flags to do URL/text cleanup only.
Pass --strip-images to also remove images (one-time; irreversible).

Overwrites each file in place after changes.
"""

import os
import sys
import glob
import argparse
import fitz  # PyMuPDF

DOCS_DIR = "docs"

# Single-token URL/word filter — any visible word token containing one of these is blanked
FILTER_KEYWORDS = (
    "youtube.com", "youtu.be", "youtube",
    "beardedboggan",
    "branchwag",
    "discord.gg", "discord.com", "discord",
    "themiddleages.net",
    "google.com",
    "thecrevaloncampaign",
)

# Block-level redaction — any text block whose content contains one of these
# substrings (case-insensitive) is fully blanked. Used for multi-line out-of-game
# notes where phrase-level search is unreliable (Unicode apostrophes, line breaks).
REDACT_BLOCK_CONTAINS = (
    # Streaming intro written before going live
    "LeAnne and we are dnd",
    "am ever Dming",
    "If streaming:",
)


def should_filter(s: str) -> bool:
    low = s.lower()
    return any(k in low for k in FILTER_KEYWORDS)


def clean_pdf(path: str, strip_images: bool) -> bool:
    doc = fitz.open(path)
    changed = False

    for page in doc:
        page_changed = False

        # --- 1. Remove link annotations pointing to filtered URLs ---
        links = page.get_links()
        for link in links:
            uri = link.get("uri", "")
            if should_filter(uri):
                page.delete_link(link)
                page_changed = True
                print(f"  removed link: {uri[:80]}")

        # --- 2. Redact visible text tokens matching filter keywords ---
        words = page.get_text("words")
        for x0, y0, x1, y1, word, *_ in words:
            if should_filter(word):
                page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(1, 1, 1))
                page_changed = True
                print(f"  redacted text: {word!r}")

        # --- 3. Block-level redaction for multi-line out-of-game notes ---
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            text_lower = text.lower()
            for marker in REDACT_BLOCK_CONTAINS:
                if marker.lower() in text_lower:
                    page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(1, 1, 1))
                    page_changed = True
                    print(f"  redacted block containing: {marker!r}")
                    break

        # --- 4. Strip all embedded images ---
        if strip_images:
            for img in page.get_images(full=True):
                xref = img[0]
                for rect in page.get_image_rects(xref):
                    page.add_redact_annot(rect, fill=(1, 1, 1))
                    page_changed = True

        if page_changed:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE if strip_images else fitz.PDF_REDACT_IMAGE_NONE)
            changed = True

    if changed:
        tmp = path + ".tmp"
        doc.save(tmp, incremental=False, encryption=fitz.PDF_ENCRYPT_NONE, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  saved: {path} ({size_mb:.1f} MB)")
    else:
        doc.close()
        print(f"  no changes: {path}")

    return changed


def main():
    parser = argparse.ArgumentParser(description="Clean PDFs in docs/")
    parser.add_argument(
        "--strip-images",
        action="store_true",
        help="Remove all embedded images from every PDF (irreversible)",
    )
    args = parser.parse_args()

    pdfs = glob.glob(f"{DOCS_DIR}/*.pdf")
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}/")
        sys.exit(0)

    if args.strip_images:
        print("Image stripping enabled — this is irreversible. Run from a git-clean state.")
        print()

    total_changed = 0
    for pdf in sorted(pdfs):
        print(f"Processing {pdf}...")
        if clean_pdf(pdf, strip_images=args.strip_images):
            total_changed += 1

    print(f"\nDone. {total_changed}/{len(pdfs)} file(s) modified.")
    if total_changed:
        print("Re-run 'make ingest ARGS=\"--fresh\"' to re-index the cleaned documents.")


if __name__ == "__main__":
    main()
