# Lab 0 — Environment Check

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/00_Environment_Check/lab0_env_check.ipynb)

**Pre-class | ~10 minutes | CPU only | No API key needed**

---

## Purpose

This is a pre-class sanity check. Run it before Day 1 so you arrive knowing your environment works. If something breaks here, it's far better to debug it now than in the middle of a lab.

---

## What You Will Verify

| Library | What It Does | Why We Need It |
|---------|-------------|----------------|
| `transformers` | Load and run HuggingFace models | Core of every lab |
| `torch` | PyTorch tensor operations | Runtime for all models |
| `sentence-transformers` | Local text embeddings | RAG labs (free, no API key) |
| `chromadb` | In-memory vector store | RAG retrieval |
| `openai` | OpenAI SDK | Communicates with any OpenAI-compatible API |
| `langchain` | LLM application framework | Orchestrates agents and memory |
| `langchain-openai` | LangChain wrapper for OpenAI | Connects LangChain to the API |
| `gradio` | Web UI framework | Lab 7 and Capstone |

The notebook installs each library, runs a minimal smoke test on each, and prints `✅ Environment check PASSED` when everything is working.

---

## Critical Points

**You do not need a GPU or an API key for this lab.** Everything runs on CPU and uses free, local models.

**The first install cell may take 2–3 minutes** in a fresh Colab instance. This is normal — Colab is installing packages from scratch.

**If a cell fails,** read the error carefully. The most common issues are:
- Version conflicts between `torch` and `bitsandbytes` — usually resolved by restarting the runtime and re-running
- `chromadb` import errors on older Python versions — Colab's default Python 3.10+ should be fine

---

## Before Moving On

Once you see `✅ Environment check PASSED`, you're ready for Day 1. You don't need to understand the code yet — that's what the rest of the labs are for.
