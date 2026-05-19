# Lab 5 — RAG Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/05_RAG_Pipeline/lab5_rag_pipeline.ipynb)

**Day 2 Morning | ~60 minutes | CPU | OpenAI API key required**

---

## Purpose

A language model's knowledge is frozen at training time. It doesn't know about your company's policies, last quarter's earnings, or the document you uploaded this morning. **Retrieval-Augmented Generation (RAG)** solves this by finding relevant information at query time and injecting it into the prompt. This lab builds the full pipeline from scratch — and then measures whether it actually works.

---

## The RAG Iceberg

The API call is the tip. Everything below the surface is what determines whether your RAG system is useful:

```
                    ┌────────────────────┐
                    │   User Question    │  ← the visible part
                    └────────┬───────────┘
                             │
          ┌──────────────────▼───────────────────────┐
          │              Retrieval                    │
          │  1. Embed the question (same model)       │
          │  2. Similarity search in vector DB        │
          │  3. Return top-k chunks                   │
          └──────────────────┬───────────────────────┘
                             │
          ┌──────────────────▼───────────────────────┐
          │              Generation                   │
          │  "Answer ONLY based on this context: …"  │
          │  + top-k chunks + user question           │
          └──────────────────┬───────────────────────┘
                             │
                    ┌────────▼───────────┐
                    │      Answer        │
                    └────────────────────┘
```

The invisible parts — chunking strategy, embedding model choice, retrieval quality, prompt design — determine 80% of your system's quality.

---

## What You Will Build

**Part A — Chunking**
You'll split documents using `RecursiveCharacterTextSplitter` and compare two configurations: small chunks (200 chars / 40 overlap) vs larger chunks (400 chars / 80 overlap). You'll see how chunk size affects what gets retrieved.

**Part B — Embeddings and Vector Store**
You'll convert each chunk to a dense vector using `all-MiniLM-L6-v2` (a free, local sentence-transformer). Store them in a ChromaDB in-memory collection. Run similarity searches and visualize the embedding space with PCA — watching related concepts cluster together.

**Part C — Retrieval and Generation**
Build a `rag(question)` function: embed the question → retrieve top-3 chunks → inject into a grounded prompt → generate an answer. Compare RAG vs no-RAG on the same question to see the grounding effect.

**Part D — Evaluation**
Run RAGAS metrics (`faithfulness`, `answer_relevancy`) using `gpt-4o` as the judge. The LLM-as-a-Judge pattern: another LLM grades whether the answer is faithful to the retrieved context and relevant to the question. If RAGAS fails, a manual rubric fallback is provided.

---

## Critical Points

**Embeddings are the semantic fingerprint of text.** Two sentences that mean the same thing produce vectors that are close together in high-dimensional space, even if they share no words. This is what makes semantic search work — the query "how do I make a model smaller?" finds chunks about "model compression" and "quantization."

**Chunking strategy is not a detail — it's a design decision.** Chunks that are too small lose context ("the parameter" — which parameter?). Chunks that are too large dilute relevance and waste context window. The right size depends on your document type and query patterns.

**We use local embeddings (`sentence-transformers`), not OpenAI embeddings.** This is intentional:
- No API key needed for embedding
- No cost per embedding call
- Works offline
- `all-MiniLM-L6-v2` (384 dimensions, 46 MB) is fast and accurate enough for most tasks

**The prompt is what enforces grounding.** The model will hallucinate if you ask it to answer freely. Adding "Answer ONLY based on the provided context. If the answer is not in the context, say so." forces the model to stay grounded. This is the single most important sentence in a RAG prompt.

**RAG does not prevent all hallucination.** If the relevant chunk isn't retrieved (retrieval failure), or the retrieved chunk is misleading (noisy data), the model can still produce a wrong answer. Evaluation is not optional — it's how you know your system works.

**LLM-as-a-Judge** uses a capable model (like `gpt-4o`) to evaluate the outputs of your system. It's not perfect, but it scales. Human evaluation is the gold standard but doesn't scale to thousands of queries.

---

## Key Terms

| Term | Definition |
|------|-----------|
| RAG | Retrieval-Augmented Generation — find relevant context at query time, inject into prompt |
| Chunk | A segment of a document. The unit of retrieval. |
| Embedding | A dense numeric vector representing the semantic meaning of a piece of text |
| Vector database | A database optimized for similarity search over embeddings |
| Cosine similarity | A measure of how aligned two vectors are. 1 = identical direction, 0 = unrelated. |
| Top-k retrieval | Return the k chunks most similar to the query embedding |
| Faithfulness | RAGAS metric: is every claim in the answer supported by the retrieved context? |
| Answer relevancy | RAGAS metric: does the answer address the question that was asked? |
| LLM-as-a-Judge | Using a capable LLM to evaluate the quality of another LLM's output |
| ChromaDB | Lightweight in-memory (or persistent) vector database, no server required |
