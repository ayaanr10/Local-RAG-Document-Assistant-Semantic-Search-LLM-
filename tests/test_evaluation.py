import unittest
from rag.evaluation import evaluate_retrieval, select_context, estimate_tokens
from rag.answering import build_prompt, answer_question, ABSTAIN
from rag.text_reader import extract_text_file
from rag.engine import clean_filename


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_rank(self):
        def search(question, top_k):
            if question == 'first':
                return [{'source':'other.pdf','page':1}, {'source':'right.pdf','page':2}]
            return [{'source':'wrong.pdf','page':1}]
        cases = [{'question':'first','expected_source':'right.pdf','expected_page':2},
                 {'question':'second','expected_source':'right.pdf','expected_page':2}]
        m = evaluate_retrieval(cases, search, top_k=4)
        self.assertEqual(m.count,2)
        self.assertEqual(m.hit_rate,.5)
        self.assertEqual(m.mrr,.25)
        self.assertEqual(m.average_first_relevant_rank,2)
    def test_empty_benchmark(self):
        m = evaluate_retrieval([], lambda *a, **k: [])
        self.assertEqual(m.count,0)
    def test_insufficient_evidence_abstains_without_api(self):
        self.assertEqual(answer_question('anything',[]).text, ABSTAIN)
    def test_context_budget_does_not_truncate(self):
        d=[{'text':'a'*140,'source':'x','page':1},{'text':'b'*140,'source':'x','page':2}]
        self.assertEqual(len(select_context(d, max_context_chars=250)),1)
    def test_prompt_has_source(self):
        p = build_prompt('What?', [{'text':'Answer here','source':'doc.pdf','page':3}])
        self.assertIn('doc.pdf, page 3',p)
        self.assertIn('untrusted',p)
    def test_token_estimator(self):
        self.assertEqual(estimate_tokens('x'*9),3)
    def test_text_reader(self):
        self.assertEqual(extract_text_file('note.md',b'Hello')[0].text,'Hello')
        with self.assertRaises(ValueError):
            extract_text_file('bad.txt',b'\xff')
    def test_filename_sanitizer(self):
        self.assertEqual(clean_filename('C:\\private\\file.pdf'),'file.pdf')

if __name__ == '__main__': unittest.main()
