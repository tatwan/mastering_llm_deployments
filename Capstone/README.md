# Capstone — End-to-End LLM Deployment System

**Day 2 Afternoon | ~3 hours | CPU (Track A) or T4 GPU (Track B)**

| Track | Notebook | Runtime |
|-------|----------|---------|
| **Track A** — Domain RAG Assistant | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Capstone/capstone_track_a.ipynb) | CPU |
| **Track B** — Fine-Tuned Showcase | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Capstone/capstone_track_b.ipynb) | **T4 GPU** ⚡ |

> **Track B:** Enable GPU before opening the notebook.
> Runtime → Change runtime type → Hardware accelerator → **T4 GPU** → Save

---

## Purpose

Every lab in this course isolated one skill. The capstone puts them all together. You'll build a complete, working system — not a demo notebook, but something with a real use case, real data, and a public URL you can share. Then you'll present it.

---

## Choose Your Track

### Track A — Domain RAG Assistant *(recommended for most students)*

**What you build:** A RAG-powered chat assistant over documents from a domain you choose. A working Gradio UI with source citations and a public URL.

**What you demonstrate:** You understand the full RAG pipeline — document loading, chunking, embedding, retrieval, generation, and grounding — and you can apply it to a new domain without hand-holding.

**Hardware:** CPU. This track works on the free Colab tier with no GPU.

**Suggested domains:**
- Your company's public documentation
- A research area you know well (Wikipedia articles, arXiv abstracts)
- A policy domain (regulations, legal docs, public standards)
- A product you use (open-source project docs, API references)

**Document loading options (covered in the template):**
- **Inline text** — paste content directly into the notebook (fastest, always works)
- **Upload PDFs** — use Colab's file picker to upload files into a `pdfs/` folder; the template loads them automatically with `PyPDFLoader`
- **Web URLs** — load public pages with `WebBaseLoader` (no file upload needed)

---

### Track B — Fine-Tuned Model Showcase *(advanced students)*

**What you build:** A before/after comparison of a base model vs a QLoRA fine-tuned version, presented in a Gradio interface that lets you run both side by side.

**What you demonstrate:** You understand quantization, LoRA adapter training, and adapter loading well enough to apply them to a new task with a dataset you choose.

**Hardware:** T4 GPU required.

---

## What Makes a Strong Capstone

The evaluation rubric has five criteria:

| Criterion | What "good" looks like |
|-----------|----------------------|
| **Working system** | App launches, handles queries end-to-end without crashing |
| **Domain relevance** | Knowledge base is genuinely relevant to the stated use case |
| **Grounding** (Track A) | Answers stay within what's in the KB; cites sources; says "I don't know" when appropriate |
| **Before/after contrast** (Track B) | The fine-tuned model demonstrably behaves differently on the target task |
| **Reflection** | You can articulate one thing that didn't work and why |

The weakest capstones tend to share one trait: they show a working demo but can't explain anything that went wrong or any limitation. The strongest ones include a failure case — a question the system gets wrong — and an explanation.

---

## Presentation Format (5 minutes)

1. **Problem (30 sec):** *"I built a [domain] assistant for [user]."*
2. **Architecture (60 sec):** Walk through the pipeline — what data, what chunking, what retrieval.
3. **Live demo (3 min):** Ask 3 questions. Include at least one the system handles well and one it doesn't.
4. **Reflection (30 sec):** What surprised you? What would you change?

You are expected to run the app live during your presentation. Have it open and warmed up before you present.

---

## Tips

**Start with the data.** The quality of your RAG system is bounded by the quality of your knowledge base. Garbage in, garbage out. Spend 20 minutes choosing and curating good source documents before writing any code.

**Test grounding early.** After you build the basic pipeline, ask a question that's *not* in your knowledge base. If the model makes something up confidently instead of saying "I don't have information about that," fix the system prompt before building the UI.

**The Gradio UI is not the hard part.** Lab 7 gave you the template. The interesting engineering is in the retrieval pipeline and the prompt design.

**Track B: Start with a small, focused task.** Fine-tuning on 10–20 high-quality examples for a narrow task (e.g., "always respond in JSON with a specific schema," "always answer medical questions with a disclaimer") shows clearer before/after contrast than a broad task.

**Document your choices.** What chunk size did you use and why? What retrieval k? What system prompt? Being able to answer these questions in your presentation is the difference between "I ran some code" and "I built a system."
