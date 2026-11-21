"""Extract selectable text from PDF pages."""

import pymupdf

from rag.chunking import PageText


def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("This file does not appear to be a valid PDF.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported in this version.")
            pages = []
            for page_num, page in enumerate(document, start=1):
                text = page.get_text("text", sort=True).strip()
                if text:
                    pages.append(PageText(page_num, text))
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise ValueError("Could not read this PDF.") from exc

    if not pages:
        raise ValueError("No selectable text found. Scanned PDFs need OCR (planned improvement).")
    return pages
