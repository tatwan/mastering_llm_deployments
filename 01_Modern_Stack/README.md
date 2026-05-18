# Lab 1 — The Modern GenAI Stack

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/01_Modern_Stack/lab1_modern_stack.ipynb)

**Day 1 Morning | ~45 minutes | CPU | OpenAI API key required**

---

## Purpose

Before writing a single line of deployment code, you need a map of the tools and why they exist. This lab orients you on the three layers of the modern GenAI stack — and introduces the single most important concept in LLM serving: the **OpenAI-compatible endpoint**.

---

## The Stack You're Learning

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

## What You Will Build

**Part A — HuggingFace Under the Hood**
You'll run GPT-2 through `pipeline()`, then peel back the abstraction to see raw tokens, logits, and top-5 next-word predictions. You'll print the memory footprint comparison across FP32, FP16, and INT4 formats — setting up the intuition for Lab 3.

**Part B — The OpenAI SDK and the `base_url` Swap**
You'll call `gpt-4o-mini`, stream a response, then do a quality comparison against `gpt-4o`. Then the key exercise: swap `base_url` to point at different providers using the exact same client code. This is the pattern that lets you develop against a local model and ship to a cloud API without changing your application code.

**Part C — LangChain Chains**
You'll build a `ChatPromptTemplate → LLM → StrOutputParser` chain. The goal is not to memorize LangChain's API, but to understand *why* orchestration layers exist: they let you compose, version, and swap components without rewriting everything.

---

## Critical Points

**The `base_url` pattern is the most important concept in this lab.** Memorize it:
```python
client = OpenAI(api_key=YOUR_KEY, base_url="https://api.openai.com/v1")
```
Every inference engine you encounter — FastAPI, vLLM, Ollama, Together AI, Groq — exposes an OpenAI-compatible endpoint. Change `base_url`, keep everything else. This is how teams swap backends without touching application code.

**`gpt-4o-mini` is your workhorse model.** It's cheap, fast, and capable enough for all labs. `gpt-4o` is used only for quality comparisons where you need to see the difference.

**HuggingFace `pipeline()` is a high-level convenience wrapper.** Under it lives `AutoTokenizer`, `AutoModel`, and a generation loop. Lab 2 takes you inside those layers.

**LangChain is an orchestration tool, not a serving tool.** It makes it easier to build chains and agents, but it doesn't run models. It calls APIs — yours or OpenAI's.

---

## Key Terms

| Term | Definition |
|------|-----------|
| `base_url` | The root URL of any OpenAI-compatible API. Swap this to change backends. |
| `pipeline()` | HuggingFace high-level API for running inference in one line |
| Logits | Raw unnormalized scores the model outputs before sampling |
| Temperature | Controls randomness in sampling. 0 = deterministic, >1 = more random |
| Chain | A LangChain construct that pipes components together: prompt → model → parser |
