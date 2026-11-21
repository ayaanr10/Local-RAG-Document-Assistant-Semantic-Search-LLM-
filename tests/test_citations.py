import unittest
from rag.citations import format_answer_citations, render_reference_key


class CitationFormattingTests(unittest.TestCase):
    def test_repeated_citation_deduplicated(self):
        raw = ('**Overview:** Useful [Reflective_Teaching-13221.pdf, p. 27].\n'
               'More evidence [Reflective_Teaching-13221.pdf, p. 27].')
        formatted, refs = format_answer_citations(raw)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]['page'], '27')
        self.assertEqual(formatted.count('atlas-citation'), 2)
        self.assertNotIn('Reflective_Teaching-13221.pdf, p. 27]', formatted)
        self.assertIn('**Overview:**', formatted)
        self.assertIn('Reflective_Teaching-13221.pdf', render_reference_key(refs))

    def test_multiple_pages_keep_distinct_numbers(self):
        _, refs = format_answer_citations('Text [abc.pdf, p. 3] and [abc.pdf, p. 4]')
        self.assertEqual(len(refs), 2)
        self.assertEqual([r['number'] for r in refs], [1, 2])

    def test_legacy_source_without_extension(self):
        opaque = 'ACFrOgD0s_S5pTRN7QPhMifbBLBwNIfaewhy5ej7tpT3go-N_FksdAElP62qOC5C2YfdhWZdYxPIABBGI7X2TifCO'
        raw = f'**Evidence:** text [{opaque}, p. 8]. More [{opaque}, p. 8].'
        body, refs = format_answer_citations(raw)
        self.assertEqual(body.count('atlas-citation'), 2)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]['page'], '8')
        self.assertIn('Uploaded document 1', render_reference_key(refs))

    def test_new_compact_source_ids_use_real_metadata(self):
        matches = [{'source': 'Sample.pdf', 'page': 3}, {'source': 'Long File Name.pdf', 'page': 6}]
        body, refs = format_answer_citations('One [S2], two [S1], three [S2]', matches)
        self.assertEqual(body.count('atlas-citation'), 3)
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0]['filename'], 'Long File Name.pdf')
        self.assertEqual(refs[1]['filename'], 'Sample.pdf')
        self.assertEqual(refs[0]['page'], '6')

    def test_invalid_source_id_not_converted(self):
        body, refs = format_answer_citations('Unsupported [S9]', [{'source': 'foo.pdf', 'page': 1}])
        self.assertEqual(refs, [])
        self.assertIn('[S9]', body)

    def test_html_escape_in_model_output(self):
        text = '<script>alert("test")</script> [foo.pdf, p. 2]'
        body, _ = format_answer_citations(text)
        self.assertNotIn('<script>', body)
        self.assertIn('&lt;script&gt;', body)

    def test_plain_answer_unchanged(self):
        body, refs = format_answer_citations('Hello **world**!')
        self.assertEqual(body, 'Hello **world**!')
        self.assertEqual(refs, [])

    def test_pdf_name_with_spaces_and_page_range(self):
        _, refs = format_answer_citations('Fact [My Notes.pdf, p. 12–13]')
        self.assertEqual(refs[0]['filename'], 'My Notes.pdf')
        self.assertEqual(refs[0]['page'], '12–13')


if __name__ == '__main__':
    unittest.main()
