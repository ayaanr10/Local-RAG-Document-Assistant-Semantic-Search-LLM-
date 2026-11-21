"""Pure-Python unit tests: no model download or API key required."""

import unittest

from rag.chunking import PageText, chunk_pages


class ChunkingTests(unittest.TestCase):
    def test_preserves_page_numbers(self):
        pages = [PageText(2, "hello there world"), PageText(7, "another page")]
        result = chunk_pages(pages, chunk_size=10, overlap=2)
        self.assertEqual([c.page_number for c in result], [2, 7])

    def test_overlap(self):
        text = " ".join(f"w{i}" for i in range(14))
        result = chunk_pages([PageText(1, text)], chunk_size=10, overlap=3)
        self.assertEqual(result[0].text.split()[-3:], result[1].text.split()[:3])
        self.assertEqual(len(result), 2)

    def test_empty_pages(self):
        self.assertEqual(chunk_pages([PageText(1, "   ")]), [])

    def test_bad_overlap(self):
        with self.assertRaises(ValueError):
            chunk_pages([PageText(1, "hello")], chunk_size=5, overlap=5)


if __name__ == "__main__":
    unittest.main()
