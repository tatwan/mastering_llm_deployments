# Lab 6 — RAG Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/06_RAG_Pipeline/lab6_rag_pipeline.ipynb)

**Day 2 Morning | ~60 minutes | CPU | OpenAI API key required**

---

## Coming from Lab 5

Lab 5 served an OpenAI-compatible API. RAG sits **in front of** that call: retrieve with MiniLM + Chroma (Lab 0), generate with `gpt-4o-mini` and a grounded prompt (Lab 2). The notebook calls OpenAI directly so the retrieval lesson stays visible. Lab 7 wraps the same pipeline in Gradio.

---

## Purpose

A language model's knowledge stops at its training date, and it never saw your documents. Ask it about them and it answers anyway. **Retrieval-augmented generation (RAG)** finds the relevant passages at question time and puts them in the prompt. This lab builds the pipeline one stage at a time, then measures whether it works.

---

## The pipeline

```
INDEX TIME (once)      load → chunk → embed → store in Chroma
QUERY TIME (per ask)   embed the question → nearest chunks → prompt + chunks → gpt-4o-mini → answer
```

The API call at the end is the part everyone sees. Chunking, the embedding model, retrieval and the prompt decide most of the quality, and each can fail on its own. The lab is built so you see several of those failures happen.

---

## What You Will Build

**Part A — Load and chunk.** Three kinds of source (inline text, two Hugging Face docs pages, the QLoRA paper as a PDF) merged into one list, then split with `RecursiveCharacterTextSplitter` at 200 and 400 characters.

**Part B — Embed and store.** `all-MiniLM-L6-v2` (free, local, 384 numbers per chunk) into a persistent Chroma folder with cosine distance. Similarity search, how to read the distances, and a PCA plot of the embedding space.

**Part C — Retrieve and generate.** A `rag()` function with a grounded prompt that cites sources. The same question about SGLang with and without retrieval: without it, `gpt-4o-mini` invents what the name stands for. Then a batch of questions where some come back "not covered", and why.

**Part D — Hybrid search.** BM25 keyword retrieval merged with the embeddings through Reciprocal Rank Fusion. It recovers the chunk Part C missed, and it shows what web-page boilerplate does to an index.

**Part E — Evaluation.** RAGAS `faithfulness` and `answer_relevancy`, with `gpt-4o` as the judge, plus a manual rubric that always works.

---

## Critical Points

**Retrieval failures look like model failures.** In this lab the QLoRA paper produces a few hundred chunks and the course notes about ten. The paper crowds the notes out of the top three, and the model correctly says it was not given the answer. Check the sources line before you touch the prompt or the model.

**The prompt decides how the model handles gaps.** "Answer only from the context" keeps it grounded. But an early version of this lab's prompt, strict rule plus one fixed refusal sentence, made `gpt-4o-mini` refuse six of nine questions it had the context for. Letting it answer part of a question and name what is missing fixed most of them, and off-topic questions are still refused.

**Chunking is a design decision.** Too small and a chunk loses its context ("the parameter", which one?). Too large and it carries text unrelated to the question. Overlap keeps sentences at the boundaries from being cut in half.

**Local embeddings are the default here on purpose.** `all-MiniLM-L6-v2` is about 90 MB, runs on CPU, needs no key and costs nothing per call. Move to a bigger model when you have evidence retrieval is missing things.

**Hybrid search covers what embeddings miss.** Embeddings handle paraphrase; BM25 handles exact tokens like `NF4`, product codes and error IDs. Reciprocal Rank Fusion merges the two rankings without a trained re-ranker.

**Clean your documents.** Web loaders keep menus and footers. Those chunks match many queries and answer none.

**RAG reduces hallucination; it does not end it.** A missed chunk or a misleading one still produces a wrong answer. Evaluation is how you find out, and an LLM judge is how you do it at scale. Human review stays the gold standard for the cases that matter.

---

## Key Terms

| Term | Definition |
|------|-----------|
| RAG | Retrieval-Augmented Generation — find relevant context at query time, inject into prompt |
| Chunk | A segment of a document. The unit of retrieval. |
| Embedding | A dense numeric vector representing the semantic meaning of a piece of text |
| Vector database | A database optimized for similarity search over embeddings |
| Cosine similarity | How aligned two vectors are: 1 = same direction, 0 = unrelated. Chroma reports cosine *distance*, which is 1 minus this. |
| Top-k retrieval | Return the k chunks most similar to the query embedding |
| Faithfulness | RAGAS metric: is every claim in the answer supported by the retrieved context? |
| Answer relevancy | RAGAS metric: does the answer address the question that was asked? |
| LLM-as-a-Judge | Using a capable LLM to evaluate the quality of another LLM's output |
| Chroma | A vector database that runs inside your Python process, in memory or in a folder on disk |
| BM25S | Fast keyword-based retrieval using the BM25 algorithm with sparse matrices |
| Hybrid search | Combining dense (semantic) and sparse (keyword) retrieval for better recall |
| Reciprocal Rank Fusion (RRF) | Score fusion method: `1/(k + rank)` per result list, summed across methods |

---

## Next

[Lab 7 — Gradio RAG App](../07_Gradio_RAG_App/README.md) — streaming chat UI on this pipeline, then a partner red-team of the public Gradio URL.
