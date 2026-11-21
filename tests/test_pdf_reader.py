"""Small integration checks for PDF extraction (PyMuPDF, no AI model needed)."""

import unittest
import pymupdf

from rag.pdf_reader import extract_pages


class PDFReaderTests(unittest.TestCase):
    def test_extract_text_and_page_number(self):
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "A useful sentence about space.")
        document.new_page()  # blank pages are skipped
        pages = extract_pages(document.tobytes())
        document.close()
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].page_number, 1)
        self.assertIn("space", pages[0].text)

    def test_invalid_pdf(self):
        with self.assertRaises(ValueError):
            extract_pages(b"not a pdf")


if __name__ == "__main__":
    unittest.main()
