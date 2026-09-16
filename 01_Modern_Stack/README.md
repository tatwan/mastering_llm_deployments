# Lab 1 — Modern Stack and Tool-Using LLMs

| Notebook | Focus | Time | Open in Colab |
| --- | --- | --- | --- |
| **Part 1A — Modern GenAI Stack** | HuggingFace internals, OpenAI SDK, `base_url` swap, LangChain chains | ~45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/01_Modern_Stack/lab1_modern_stack.ipynb) |
| **Part 1B — Tool Calling, ReAct, and SQL Agents** | Tool calling, ReAct as a loop, SQL agent from scratch | ~45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/01_Modern_Stack/lab1_part2_tools_react_sql_agent.ipynb) |

**Day 1 Morning | CPU | `OPENAI_API_KEY` required** (instructor provides; store as a Colab Secret)

---

## Coming from Lab 0

Lab 0 proved the libraries load and showed tokens, embeddings, and a tiny retrieval. Lab 1 puts a **model** on those tokens, then switches to the hosted API every later lab uses.

You need the OpenAI key from here on (Labs 1A, 1B, 2, 5, 6, 7).

---

## Purpose

Before writing deployment code, you need two mental models:

1. **LLM as inference engine:** prompt in, text out. This is Part 1A.
2. **LLM as decision-maker:** the model requests actions through tools; your code runs them; the model answers from the observation. This is Part 1B.

The through-line of the whole course lives in 1A: **one OpenAI client, swap `base_url`.** Lab 5 is that idea with *your* FastAPI server. Groq, vLLM, and Ollama are the same idea with someone else's.

---

## The stack you are learning

```
┌─────────────────────────────────────────────────┐
│  Orchestration     LangChain · LlamaIndex        │  chains, agents, RAG workflows
├─────────────────────────────────────────────────┤
│  Serving API       OpenAI SDK · FastAPI          │  the "one client" pattern
├─────────────────────────────────────────────────┤
│  Models & Hubs     HuggingFace Transformers      │  weights, tokenizers, pipelines
└─────────────────────────────────────────────────┘
```

---

## Part 1A — What you will build

**Part A — HuggingFace under the hood.** GPT-2 on CPU through `pipeline()`, then `AutoTokenizer` + `generate()`, then a raw forward pass (logits → top-5 next tokens). A memory table (FP32 / FP16 / INT4) sets up Lab 4.

**Part B — OpenAI SDK and the `base_url` swap.** Call `gpt-4o-mini`, stream, compare to `gpt-4o`. Then the key move: the same `OpenAI(...)` constructor pointed at another backend. Optional Groq if a `GROQ_API_KEY` secret exists — never paste a key into a cell.

**Part C — LangChain.** `ChatPromptTemplate | ChatOpenAI | StrOutputParser`. The goal is not to memorize LangChain. The goal is: same `base_url`, higher-level app structure.

## Part 1B — What you will build

**Part A — One tool.** `estimate_model_memory` (the Lab 1A arithmetic). You watch `finish_reason` change from `stop` to `tool_calls`, run the function yourself, send a `role=tool` observation, and get a grounded answer.

**Part B — One HTTP tool.** Open-Meteo weather (no extra key). Same two-round loop; only the tool changes. If the network is blocked, the function returns mock JSON so the lab continues.

**Part C — ReAct as a loop.** You already ran ReAct in A–B. This part is a *short* contrast: old text-parsed `Action:` lines vs modern JSON `tool_calls`. Bonus 03 is the longer from-scratch text loop if you want it later.

**Part D — SQL agent from scratch.** SQLite table of classroom benchmark numbers (Qwen sizes you will load in Labs 3–4). A `SELECT`-only allowlist. Natural language in, gated SQL, explanation out.

**Exercise.** A TODO cell for `get_efficiency_score` — a second tool on the same database.

---

## Critical points

**Memorize this constructor:**

```python
client = OpenAI(api_key=YOUR_KEY, base_url="https://api.openai.com/v1")
```

Every inference engine in this course — FastAPI (Lab 5), vLLM (concepts), Ollama, Groq, LiteLLM (Bonus 04) — speaks this protocol. Change `base_url`. Keep the rest.

**`gpt-4o-mini` is the workhorse.** Cheap, fast, good enough. `gpt-4o` is only for quality comparisons and Lab 6's judge.

**`pipeline()` is a convenience wrapper.** Under it: tokenizer, model, generation loop. Lab 3 stays inside those layers with Qwen2.5-0.5B.

**LangChain orchestrates. It does not serve.** It calls APIs — yours or OpenAI's.

**The model never gets a database connection.** You expose `run_sql`. You validate. That boundary is the product.

---

## Key terms

| Term | Definition |
|------|-----------|
| `base_url` | Root URL of an OpenAI-compatible API. Swap this to change backends. |
| `pipeline()` | HuggingFace high-level inference helper |
| Logits | Raw scores over the vocabulary, before softmax / sampling |
| Temperature | Sampling randomness. 0 ≈ greedy; higher ≈ more random (Lab 3) |
| Chain | Prompt → model → parser, piped together |
| Tool / function calling | Model returns structured `tool_calls`; your code executes them |
| ReAct | Reason + Act loop: thought, action, observation, answer |
| SQL agent | An agent that answers by generating and running *constrained* SQL |

---

## Before moving on

1A is done when you have streamed gpt-4o-mini and understood that `base_url` is the only line that must change to retarget a backend.

1B is done when you have seen `finish_reason='tool_calls'`, a SQL agent that reads *your* table (not internet guesses), and you know why regex-ReAct is a teaching device rather than a production plan.

Next: [Lab 2 — Prompting](../02_Prompting/README.md). Same client, now the prompt is the product — including how it fails under injection.
