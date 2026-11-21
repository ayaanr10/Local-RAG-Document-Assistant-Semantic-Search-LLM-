"""Upload -> parse -> chunk -> embed -> Chroma; query -> embed -> retrieve.

Chroma and the embedding model are imported lazily, keeping non-ML unit tests
fast and avoiding downloads until retrieval is actually used.
"""
from __future__ import annotations

import hashlib
import os
from functools import cached_property
from pathlib import Path
import re

from rag.chunking import chunk_pages
from rag.pdf_reader import extract_pages
from rag.text_reader import extract_text_file

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MAX_PDF_BYTES = 20 * 1024 * 1024


class FastEmbedAdapter:
    """Small local ONNX-based MiniLM backend for resource-limited hosting."""
    def __init__(self) -> None:
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=MODEL_NAME)

    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        import numpy as np
        vectors = np.asarray(list(self.model.embed(texts)), dtype=np.float32)
        if normalize_embeddings and vectors.size:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors /= np.maximum(norms, 1e-12)
        return vectors


def clean_filename(filename: str) -> str:
    """Reduce user-supplied filenames to a safe, readable basename."""
    base = filename.replace('\\', '/').split('/')[-1]
    base = re.sub(r'[^a-zA-Z0-9_. -]', '_', base).strip(' .')
    suffix = Path(base).suffix.lower()
    if suffix in ('.pdf', '.md', '.txt', '.markdown'):
        return (base[:-len(suffix)][:90-len(suffix)] + suffix) or 'untitled'
    return base[:90] or 'untitled'


class RAGEngine:
    def __init__(self, db_path: str = 'data/chroma', collection_name: str = 'rag_docs') -> None:
        import chromadb
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=None,
            metadata={'hnsw:space': 'cosine'},
        )

    @cached_property
    def encoder(self):
        if os.getenv('EMBEDDING_BACKEND', 'sentence-transformers') == 'fastembed':
            return FastEmbedAdapter()
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)

    def list_documents(self) -> list[dict]:
        res = self.collection.get(include=['metadatas'])
        output = {}
        for meta in res.get('metadatas') or []:
            if not meta:
                continue
            key = meta['file_hash']
            item = output.setdefault(key, {
                'file_hash': key, 'name': meta['source'],
                'chunks': 0, 'type': meta.get('file_type', 'PDF'),
            })
            item['chunks'] += 1
        return sorted(output.values(), key=lambda d: d['name'].lower())

    def stats(self) -> dict:
        docs = self.list_documents()
        return {'documents': len(docs), 'chunks': sum(d['chunks'] for d in docs)}

    def add_document(self, filename: str, raw: bytes) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix not in ('.pdf', '.txt', '.md', '.markdown'):
            raise ValueError('Upload a PDF, TXT or Markdown file.')
        if not raw:
            raise ValueError('The uploaded file is empty.')
        if suffix == '.pdf' and len(raw) > MAX_PDF_BYTES:
            raise ValueError('PDFs must be at most 20 MB.')
        if suffix != '.pdf' and len(raw) > 5 * 1024 * 1024:
            raise ValueError('Text/Markdown files must be at most 5 MB.')
        file_hash = hashlib.sha256(raw).hexdigest()
        source = clean_filename(filename)
        if any(d['file_hash'] == file_hash for d in self.list_documents()):
            return {'name': source, 'chunks': 0, 'status': 'already indexed'}
        pages = extract_pages(raw) if suffix == '.pdf' else extract_text_file(filename, raw)
        chunks = chunk_pages(pages, chunk_size=220, overlap=50)
        if not chunks:
            raise ValueError('No searchable text could be extracted.')
        try:
            for offset in range(0, len(chunks), 48):
                batch = chunks[offset:offset + 48]
                texts = [chunk.text for chunk in batch]
                vectors = self.encoder.encode(texts, normalize_embeddings=True).tolist()
                ids = [f'{file_hash[:24]}-p{c.page_number}-c{c.chunk_number}' for c in batch]
                metas = [
                    {'source': source, 'page': c.page_number, 'file_hash': file_hash,
                     'file_type': suffix.lstrip('.').upper()}
                    for c in batch
                ]
                self.collection.upsert(ids=ids, documents=texts,
                                       embeddings=vectors, metadatas=metas)
        except Exception:
            self.delete_document(file_hash)
            raise
        return {'name': source, 'chunks': len(chunks), 'status': 'indexed'}

    def add_pdf(self, filename: str, raw: bytes) -> dict:
        if Path(filename).suffix.lower() != '.pdf':
            raise ValueError('Expected a .pdf file')
        return self.add_document(filename, raw)

    def delete_document(self, file_hash: str) -> None:
        self.collection.delete(where={'file_hash': file_hash})

    def search(self, question: str, top_k: int = 4, max_distance: float | None = None,
               file_hash: str | None = None) -> list[dict]:
        if not question.strip() or top_k < 1 or not self.collection.count():
            return []
        vec = self.encoder.encode([question], normalize_embeddings=True).tolist()
        query_args = {
            'query_embeddings': vec,
            'n_results': min(top_k, self.collection.count()),
            'include': ['documents', 'metadatas', 'distances'],
        }
        if file_hash:
            query_args['where'] = {'file_hash': file_hash}
        try:
            result = self.collection.query(**query_args)
        except Exception:
            if file_hash and not any(d['file_hash'] == file_hash for d in self.list_documents()):
                return []
            raise
        texts = (result.get('documents') or [[]])[0]
        metas = (result.get('metadatas') or [[]])[0]
        distances = (result.get('distances') or [[]])[0]
        output = []
        for txt, meta, dist in zip(texts, metas, distances):
            if max_distance is not None and float(dist) > max_distance:
                continue
            output.append({
                'text': txt, 'source': meta['source'], 'page': int(meta['page']),
                'distance': float(dist), 'file_hash': meta['file_hash'],
            })
        return output
