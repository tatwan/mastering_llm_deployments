# Lab 3 — Inspect & Talk to Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/03_Inspect_Chat/lab3_inspect_chat.ipynb)

**Day 1 Morning | ~45 minutes | CPU | No API key needed**

---

## Purpose

You can call an LLM API without knowing how it works. But when something goes wrong in production — the model ignores instructions, the conversation loses context, costs spike unexpectedly — understanding the internals is what lets you diagnose and fix it. This lab builds that foundation.

---

## What You Will Build

**Part A — Model Architecture Inspection**
You'll load `Qwen2.5-0.5B-Instruct` and print its configuration: hidden size, number of layers, attention heads, vocabulary size, and context window. Then you'll walk the named modules to see the actual layer names. The goal is to connect the abstract "transformer" concept to a real, inspectable Python object.

**Part B — Generation Controls**
You'll run inference using `apply_chat_template` (the correct way) and compare the actual generated outputs to passing a raw string. You'll tune temperature and observe the difference between greedy decoding (deterministic) and sampling (probabilistic). You'll see firsthand why the same prompt gives different outputs at different temperatures.

**Part C — Multi-Turn Chat Session**
You'll build a `ChatSession` class that maintains conversation history, estimates token count, and warns when approaching the context window limit. You'll run a 3-turn conversation and inspect the full prompt being sent each time — which is the entire history, not just the latest message.

---

## Critical Points

**Tokens are not words.** The model never sees text — it sees a sequence of integer IDs. "deployment" might be one token or three depending on the tokenizer's vocabulary. This affects:
- **Cost** — APIs charge per token, not per word
- **Context limits** — "128K context window" means 128,000 tokens, roughly 100,000 words
- **Latency** — more tokens in = more compute

**Chat templates matter.** A chat model was fine-tuned with a specific prompt format: `<|system|>`, `<|user|>`, `<|assistant|>` markers (varies by model). Passing a raw string bypasses this format and degrades output quality. Always use `tokenizer.apply_chat_template()`.

**Multi-turn context is manual.** Unlike a stateful chat UI, the API is stateless. Every request must include the *full conversation history*. The model has no memory between calls — you maintain the history and send it every time. This means context grows with every turn, and eventually hits the limit.

**Smaller models trade capability for speed.** `Qwen2.5-0.5B` runs on CPU in reasonable time. It will make mistakes a larger model wouldn't. That's intentional — Lab 4 shows how to quantize and fine-tune to compensate.

---

## Key Terms

| Term | Definition |
|------|-----------|
| Tokenization | Converting text to integer IDs the model can process |
| Context window | Maximum number of tokens a model can process in one call (input + output) |
| Temperature | Sampling randomness. 0 = always pick the top token. Higher = more variety. |
| `apply_chat_template` | Formats conversation history into the model's expected input format |
| Greedy decoding | Always picking the highest-probability next token. Fast, deterministic, often repetitive. |
| Top-p (nucleus) sampling | Sample from the smallest set of tokens whose cumulative probability exceeds p |
| Attention heads | Parallel attention mechanisms that each learn to focus on different relationships in the input |

---

## Before This Lab

No API key needed — this lab uses a small local model (`Qwen2.5-0.5B-Instruct`, ~900 MB). The first download takes 2–3 minutes on a fresh Colab instance. Run the install cell and let it download while you read through the notebook.
