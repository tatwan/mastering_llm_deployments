# Lab 0 — Environment Check

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/00_Environment_Check/lab0_env_check.ipynb)

**Pre-class | ~10 minutes | CPU only | No API key needed**

---

## Purpose

This is a pre-class sanity check, not a modeling lab. Run it before Day 1 so you arrive knowing the runtime works.

You will install the libraries this course uses all week, then run a **tiny smoke test** for each layer of the stack. If something breaks here, debug it now — not in the middle of Lab 1.

You do **not** need to master the APIs yet. Read the short "why" above each cell, run the cell, and confirm the output looks like the checkpoint.

---

## What You Will Verify

| Check | Library | What you will see | Where it shows up later |
|-------|---------|-------------------|-------------------------|
| Tokenizer | `transformers` | A sentence split into token IDs | Labs 1A, 3, 4 |
| Embeddings | `sentence-transformers` | A 384-number vector | Labs 6–9, 11–12 |
| Vector store | `chromadb` | Nearest document for a query | Labs 6, 7 |
| Hosted-LLM client | `openai`, `langchain-openai` | Imports succeed (no API call) | Labs 1, 2, 5–7 |
| UI toolkit | `gradio` | Import succeeds | Lab 7 |
| Tensor runtime | `torch` | Version print; CUDA will be `False` on CPU | Labs 3–4 |

---

## How to Run

**Colab (recommended):** click the badge above. Runtime → CPU is enough.

**Local:** from the repo root, `uv pip install -r requirements.txt` (or `pip`), then open this notebook in Jupyter. The install cell also works locally; it installs into the kernel's Python.

The first Colab install takes 2–3 minutes. The MiniLM embedding model is an ~80 MB download the first time only.

---

## Critical Points

**No GPU and no API key for this lab.** Everything runs on CPU with free local files from Hugging Face Hub.

**You will need accounts later — just not today.**

| When | What | How you store it |
|------|------|------------------|
| Day 1, Lab 1A | `OPENAI_API_KEY` (instructor provides) | Colab Secret named `OPENAI_API_KEY` |
| Day 1 PM, Lab 4 | T4 GPU | Runtime → Change runtime type → T4 GPU |
| Day 2, Lab 5 | Free ngrok authtoken | Colab Secret named `NGROK_AUTH_TOKEN` |

Never paste a key into a notebook cell.

**If a cell fails,** read the error, then use the troubleshooting table at the bottom of the notebook. The usual fix is: re-run the install cell, or Runtime → Restart session and start from the top.

---

## Before Moving On

You are done when the last code cell prints `Environment check PASSED`.

Next: [Lab 1A — Modern Stack](../01_Modern_Stack/README.md). That is where you look inside HuggingFace, call `gpt-4o-mini`, and learn the `base_url` swap this course is built on.
