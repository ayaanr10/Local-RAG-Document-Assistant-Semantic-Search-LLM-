# Atlas RAG — Document Intelligence Dashboard

A full-stack-in-one-process Streamlit RAG app to **index, search, and ask grounded questions about PDFs, Markdown, and TXT files**, with source inspection and transparent retrieval-quality evaluation.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-brightgreen) ![Embedding](https://img.shields.io/badge/Embedding-MiniLM-9cf) ![Vector DB](https://img.shields.io/badge/Vector%20DB-Chroma-orange)

## Core workflow

```
INDEXING (once per document)
PDF/TXT/Markdown -> extract text (PyMuPDF or UTF-8 reader)
  -> 220-word windows / 50-word overlap
  -> SentenceTransformer all-MiniLM-L6-v2 -> 384D embeddings
  -> ChromaDB cosine index + original text + filename/page number

QUESTION ANSWERING (for each question)
Question -> same embedding model -> cosine-nearest ChromaDB top-K
  -> select complete chunks within user-controlled context budget
  -> Gemini grounded prompt (optional) -> answer + inspectable passages
```

## Dashboard features

- **Ask documents:** clean chat interface with collapsible supporting passages and retrieval details.
- **Knowledge Base:** upload/manage PDF, Markdown and UTF-8 TXT; duplicate detection by SHA-256, delete indexed documents.
- **Evaluation Lab:** upload labelled CSV (`question,expected_source,expected_page`); compute Hit Rate@K and Mean Reciprocal Rank (MRR), see first relevant rank for each question.
- **Analytics:** query count, retrieval latency and optional token details (available in a collapsed panel or Evaluation tab).
- **Controls:** advanced settings are tucked into a collapsed sidebar panel (top-K, context budget and optional distance filtering).
- **Retrieval-only:** a Gemini API key is optional; without it, the application shows source passages but cannot synthesize answers.
- **Demo data:** two self-contained sample documents and an example evaluation benchmark CSV.

## Local installation (Windows, VS Code)

Requirements: 64-bit Python 3.11 or 3.12 is recommended; stable internet access for dependency and first model download. Python 3.13 may work depending on wheel availability.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If Python 3.11 is not installed, use `py -m venv .venv` with your existing supported Python interpreter.

Open the local URL printed in the terminal (normally `http://localhost:8501`). Expand **Demo tools** in the sidebar and click **Load sample documents** first; embedding model weights download on first indexing. Then try **What is retrieval-augmented generation?**

### Optional Gemini answer generation

1. Copy `.env.example` to `.env`.
2. Put your own `GEMINI_API_KEY=...` in `.env` and choose a supported `GEMINI_MODEL` from Google AI Studio.
3. Restart the app. **Never upload `.env` to GitHub.**

With Gemini enabled, selected source passages are sent to Google's model API. Without a key the system still retrieves evidence locally.

## Automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Includes tests for PDF extraction, chunk overlap, TXT/Markdown parsing, prompt structure, empty-evidence handling, token estimates, and retrieval evaluation calculations. **Unit-test success does not prove full end-to-end performance.** Real MiniLM/Chroma/Gemini integration must be smoke-tested in the target environment.

## How to evaluate retrieval

1. Expand **Demo tools** and click **Load sample documents**.
2. Go to **Evaluation**, download the example CSV and upload that same file.
3. Click **Run retrieval benchmark**.
4. Examine Hit Rate@K and MRR. With additional documents, curate a new set of questions and labels matching your collection.

The benchmark requires **manually verified source/page labels**. It measures evidence retrieval, *not generated answer accuracy or faithfulness*. Do not claim an empirical score until you have actually executed it.

## Deployment options

### A. Streamlit Community Cloud

1. Push this folder's **contents** to a GitHub repository (with `app.py` at the repository root).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app** -> GitHub repo -> `app.py` -> Deploy.
3. Add `GEMINI_API_KEY` to the app's secrets/environment configuration if generation is desired.
4. Set `PUBLIC_DEMO=1` in the app environment, if supported. Without it, all visitors share the same local index; do not allow public uploads in that configuration.
5. Test with sample documents. Community Cloud imposes resource limits, and the SentenceTransformer + Chroma stack may exceed them. Consider a larger hosted instance or a smaller inference backend if this happens.

### B. Render (Blueprint included)

1. Push the project to GitHub.
2. In [Render](https://dashboard.render.com/), create a **Blueprint** from the repository. The included `render.yaml` specifies a Python web service, build/start commands, a health check, and `PUBLIC_DEMO=1`.
3. Configure `GEMINI_API_KEY` in Render's server-side Environment settings, or omit it for retrieval-only mode.
4. Wait for a successful build and verify document indexing, query retrieval, and API generation.

**Caveat:** Render Free web services cannot attach persistent disks; uploads/indexes can disappear after restarts/redeploys. This is a public **demo**, not durable production document storage. The Render Blueprint selects **FastEmbed + ONNX** and installs `requirements-render.txt`, avoiding the heavyweight PyTorch runtime. It still needs a live memory/integration test, and the free-tier CPU/RAM may remain insufficient for some workloads.

### Privacy and security

The `PUBLIC_DEMO=1` setting creates a random isolated Chroma collection per browser session, reducing cross-session exposure. This is **not strong multi-tenant authorization**: data can remain on the server after a visitor disconnects, and a public host is not suitable for sensitive documents. Users should upload only public/test materials. Production use requires authentication, deletion/retention policy, upload controls, storage isolation, rate limiting, and security review.

### Persistence and embedding identity

- Local Chroma database: `data/chroma/` (not committed to GitHub).
- The embedding model and inference backend must be stable for a given index. Switching encoder implementations means rebuilding the index.
- Free hosted disk storage is often ephemeral; don't promise persistent knowledge to public users.

## Repository structure

```
app.py                     # Streamlit dashboard
rag/pdf_reader.py          # PDF page extraction
rag/text_reader.py         # UTF-8 TXT and Markdown ingestion
rag/chunking.py            # Source-aware overlapping chunks
rag/engine.py              # Chroma, embedding (Sentence Transformers / FastEmbed), retrieval
rag/answering.py           # Grounded Gemini generation, retrieval-only mode
rag/evaluation.py          # Hit Rate@K, MRR and context budget
samples/                   # Public, synthetic demo knowledge base
eval/benchmark_example.csv # Labelled retrieval benchmark example
tests/                     # Unit tests
render.yaml                # Render deployment Blueprint (FastEmbed mode)
requirements-render.txt    # Lean ONNX/CPU deps for hosting
.github/workflows/tests.yml# GitHub CI
```

## Honest status and limitations

**Implemented in source:** simplified dashboard, PDF/TXT/Markdown ingestion, MiniLM embeddings, Chroma cosine search, duplicate management, retrieval inspection, grounded Gemini prompt, optional API mode, token estimates, retrieval evaluation and CI configuration.

**Tested in this package:** unit tests + Python syntax compilation.

**Not yet confirmed:** real end-to-end MiniLM/Chroma indexing, paid/free host memory compatibility, Gemini key/API call, public URL. These require running the app with actual dependencies and a configured deployment account.

**Future improvements:** OCR for scanned PDFs, semantic/sentence-aware chunking, cross-encoder reranking, robust out-of-scope detection, verified claim citations, answer-faithfulness evaluation, authenticated persistent storage.

### Answer citations and display
Generated citations such as `[notes.pdf, p. 12]` are shown as compact teal `[1]` markers. A numbered reference key appears beneath the answer; use **View supporting passages** to inspect the actual retrieved text and page metadata. These citations are provided by the LLM and are not independently validated against the source contents.

## Public demo safeguards (not production security)

When `PUBLIC_DEMO=1`, a browser session may index up to five documents,
ask up to twelve questions, and run up to 25 evaluation cases. These are
client-session limits, **not a server-side authenticated quota**; a public
service with an API key can still be abused by determined visitors.
For a production release add authentication, server-side rate limiting,
resource quotas, retention/cleanup policies, and durable isolated storage.

If the host has no key, the demo still shows retrieved supporting passages.
The deployed model is `gemini-3.6-flash` when an API key is configured.

When using Render Free, allow for cold starts and possible out-of-memory
errors from model downloads or indexing workloads. This package has not
been end-to-end smoke-tested on Render.
