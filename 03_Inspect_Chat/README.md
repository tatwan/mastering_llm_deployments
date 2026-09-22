# Lab 3 — Inspect & Talk to Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/03_Inspect_Chat/lab3_inspect_chat.ipynb)

**Day 1 Morning | ~60 minutes | CPU | No API key needed**

---

## Coming from Lab 2

Lab 2 prompted a **hosted** model (`gpt-4o-mini`) through the OpenAI client. You could not see inside it. Lab 3 puts a **local** instruct model on your CPU and takes it apart: the config, the weight tensors, the memory it allocates while generating, and the prompt string it actually receives.

No `OPENAI_API_KEY` today. Lab 4 is the next notebook that needs special hardware (T4).

---

## Purpose

You can call an LLM API without knowing how it works — right up until something goes wrong. The model ignores your system prompt. The conversation loses the thread. Costs climb for no obvious reason. Your GPU runs out of memory at eleven concurrent users when you budgeted for forty.

Every one of those is diagnosable if you know what the model is doing, and mystifying if you do not. This lab builds that foundation, and it does it with numbers you produce yourself rather than claims you take on faith.

---

## What You Will Build

**Part A — What is inside the model (~20 min)**
Attention explained as three vectors per token (query, key, value), then heads, then the history that runs MHA → MQA → GQA. You load `Qwen2.5-0.5B-Instruct`, read its config, print one decoder layer, and inspect the attention projection shapes — where you find `q_proj` producing 896 numbers and `k_proj` producing 128. That gap *is* Grouped Query Attention, visible in the weights. Closes with a GPT-2 vs Qwen2.5 config comparison: GPT-2 has no `num_key_value_heads` field at all.

**Part B — The KV cache (~12 min)**
Why generation would be quadratic without a cache, and what gets stored instead. You run a forward pass, inspect the real cache tensor (`(1, 2, 5, 64)` — that `2` is GQA in memory), watch it grow with sequence length, and time generation with `use_cache=True` against `use_cache=False`. The gap is 15–20× depending on the machine. You then turn bytes-per-token into a concurrent-user estimate for a 16 GB GPU, GQA versus MHA.

**Part C — Generation controls (~15 min)**
Tokenization, then the actual next-token probability distribution — top 5 candidates with their probabilities. Then the same logits at three temperatures so you see the distribution change shape before you see any generated text. Greedy run twice prints `Identical? True`; sampling run twice prints `False`. Then top-p, and why chat templates are a contract you cannot skip.

**Part D — Multi-turn chat (~13 min)**
You build one conversation turn by hand in four small cells, then do the identical four steps again for turn two. Only then does `ChatSession` appear — as a compression of code you already wrote, with an argument for why a class rather than a function. You print the full prompt behind a one-sentence answer, measure context pressure, and write the sliding-window `trim` yourself.

---

## Critical Points

**GQA decides your serving capacity.** `num_key_value_heads` is the first number to check when comparing open models for self-hosting. Fourteen query heads sharing two key/value heads makes the KV cache seven times smaller, which is the difference between a few dozen concurrent users and a handful on the same GPU. Benchmark scores tell you whether a model is good; this tells you whether you can afford to run it.

**The KV cache is load-bearing, not an optimization.** You will measure generation with it and without it. Memory buys compute, and the memory scales with users × context length. Every serving stack you meet later (vLLM in Lab 5) is fundamentally a KV cache manager.

**Tokens are not words.** The model sees integers. `' Paris'` and `'Paris'` are different tokens. `'巴黎'` is a single token, which is why Qwen's vocabulary is three times GPT-2's. This drives cost, context limits, and latency.

**Determinism is a product decision.** Greedy decoding is reproducible and therefore testable, cacheable, and debuggable. Sampling is not. For anything a machine has to parse downstream, start at `do_sample=False`.

**Chat templates are a contract.** An instruct model was fine-tuned on a specific format with special tokens. Send a raw string and you often get plausible-looking output that ignores your system prompt — a bug that survives testing and surfaces in production. The lab also shows that Qwen's template **injects a default system message** ("You are Qwen, created by Alibaba Cloud...") when you do not supply one, which is a real source of confusing bug reports.

**A 0.5B model gets things wrong, loudly.** It misdefines quantization in Part D and ignores a two-sentence limit in its own system prompt. The notebook calls this out where it happens. Students are here to study mechanism — shapes, memory, prompt format — not to learn the subject matter from the model. Larger models fail the same ways more subtly.

**Multi-turn memory is an illusion you pay for.** The API is stateless. You resend the entire history every turn. When it overflows, most stacks truncate silently: HTTP 200s in your logs, users saying the bot forgot something.

---

## Key Terms

| Term | Definition |
|------|-----------|
| Query / Key / Value | The three vectors each token produces. Query is what it seeks, key is what it advertises, value is what it contributes when attended to. |
| Attention head | One parallel set of Q/K/V projections. Each head learns different token relationships. |
| MHA | Multi-Head Attention. Every head has its own keys and values. The 2017 original; GPT-2 uses it. |
| MQA | Multi-Query Attention (Shazeer, 2019). All query heads share one key/value head. Cheapest, some quality loss. |
| GQA | Grouped Query Attention (Ainslie et al., Google, EMNLP 2023). Query heads share key/value heads in groups. The modern default. |
| KV cache | Stored keys and values for tokens already processed, so they are not recomputed each generation step. The main memory bottleneck in serving. |
| Context window | Maximum tokens in one call, input and output combined. |
| Tokenization | Converting text to the integer IDs the model actually processes. |
| Logits | Raw per-token scores before the softmax turns them into probabilities. |
| Temperature | Divides the logits before the softmax. Low sharpens the distribution, high flattens it. |
| Top-p (nucleus) | Keeps only the smallest set of tokens whose probabilities sum to p, then samples from those. |
| Greedy decoding | Always take the highest-probability token. Deterministic and reproducible. |
| `apply_chat_template` | Formats a message list into the exact string the model was fine-tuned on. |
| Sliding window | Keeping the system prompt plus the last N turns and dropping the rest. |

---

## Before This Lab

No API key. The lab downloads `Qwen/Qwen2.5-0.5B-Instruct` (about **1 GB**) on first run, plus GPT-2's config file (a few KB, no weights). Run the install cell and read Part A's attention section while the weights download.

One cell in Part B deliberately takes 30–90 seconds — it generates without the KV cache so you can time the difference. That wait is the point.

---

## Instructor Note

Part A is written to follow a live walkthrough of the [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) on GPT-2. The notebook has a callout pointing students at Query/Key/Value columns and the attention score lines, then cell A8 compares GPT-2's config against Qwen2.5's so the browser demo and the notebook close the loop on the same model.

---

## Next

[Lab 4 — Quantize + LoRA](../04_Quantize_LoRA/README.md) — **T4 GPU required**. Same family (Qwen2.5-1.5B), NF4 quantization, a tiny LoRA adapter. The INT4 estimate from cell A3 and the capacity math from cell B4 become measurements on real hardware. Enable the GPU **before** you start that notebook.
