# Retrieval-Augmented Generation — Sample Knowledge Base

Retrieval-Augmented Generation (RAG) is a technique that retrieves relevant passages from an external knowledge source and supplies those passages to a language model while generating a response. It does not require retraining the language model when the document collection changes.

The indexing pipeline extracts document text, divides it into smaller chunks, creates embeddings, and stores chunks and source metadata in a vector database. An embedding is a numerical vector intended to represent important characteristics of the text. Sentence Transformers are models that can create dense semantic embeddings.

The query pipeline embeds the user's question using the same embedding model and performs a similarity search against stored document vectors. The database returns the top-K nearest chunks. The application combines the question and retrieved passages in a grounded prompt, then asks an LLM to generate a supported answer.

Retrieval failures can cause a RAG system to miss relevant evidence. Generation failures can produce hallucinations even when relevant evidence is present. Source citations and tests can help expose these problems, but citations themselves must be verified.

Hit Rate at K measures the fraction of questions for which at least one relevant passage appears among the top K retrieved results. Mean Reciprocal Rank (MRR) averages the inverse rank of the first relevant retrieved passage for each question, using zero when none is retrieved.
