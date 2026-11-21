import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from rag.engine import FastEmbedAdapter


class FakeEmbedding:
    def __init__(self, model_name):
        self.name=model_name
    def embed(self, texts):
        return (np.asarray([3.0, 4.0, 0.0],dtype=np.float32) for _ in texts)


class BackendTests(unittest.TestCase):
    def test_onnx_adapter_shapes_and_normalization(self):
        fake_module = types.SimpleNamespace(TextEmbedding=FakeEmbedding)
        with patch.dict(sys.modules, {'fastembed': fake_module}):
            model = FastEmbedAdapter()
            vectors = model.encode(['hello','world'],normalize_embeddings=True)
        self.assertEqual(vectors.shape,(2,3))
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])),1.0,places=5)

if __name__ == '__main__': unittest.main()
