# Lab 1 — Modern Stack and Tool-Using LLMs

| Notebook | Focus | Time | Open in Colab |
| --- | --- | --- | --- |
| **Part 1 — Modern GenAI Stack** | HuggingFace internals, OpenAI SDK, `base_url` swap, LangChain chains | ~45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/01_Modern_Stack/lab1_modern_stack.ipynb) |
| **Part 2 — Tool Calling, ReAct, and SQL Agents** | Tool/function calling, ReAct mental model, SQL agent from scratch | ~45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/01_Modern_Stack/lab1_part2_tools_react_sql_agent.ipynb) |

**Day 1 Morning | CPU | OpenAI API key required**

---

## Purpose

Before writing deployment code, you need two mental models:

1. **LLM as inference engine:** prompt in, text out. This is Part 1.
2. **LLM as decision-maker:** the model can request actions through tools/functions, observe the results, and answer with grounded data. This is Part 2.

Together, these two notebooks orient you on the modern GenAI stack and the agentic pattern that sits on top of it.

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

## Part 1 — What You Will Build

**Part A — HuggingFace Under the Hood**
You'll run GPT-2 through `pipeline()`, then peel back the abstraction to see raw tokens, logits, and top-5 next-word predictions. You'll print the memory footprint comparison across FP32, FP16, and INT4 formats — setting up the intuition for Lab 4.

**Part B — The OpenAI SDK and the `base_url` Swap**
You'll call `gpt-4o-mini`, stream a response, then do a quality comparison against `gpt-4o`. Then the key exercise: swap `base_url` to point at different providers using the exact same client code. This is the pattern that lets you develop against a local model and ship to a cloud API without changing your application code.

**Part C — LangChain Chains**
You'll build a `ChatPromptTemplate → LLM → StrOutputParser` chain. The goal is not to memorize LangChain's API, but to understand *why* orchestration layers exist: they let you compose, version, and swap components without rewriting everything.

## Part 2 — What You Will Build

**Part A — Tool Calling**
You'll expose a small Python function to the model as a structured tool. The model will decide when to call it, your code will execute it, and the model will use the result to answer.

**Part B — ReAct**
You'll connect tool calling to the ReAct pattern: reasoning plus acting. The notebook explains how older text-parsed ReAct loops relate to modern structured function calling.

**Part C — SQL Agent From Scratch**
You'll build a practical SQL agent without LangChain or another framework. The LLM receives a database schema and a `run_sql` tool, asks to run a safe `SELECT`, observes the rows, and explains the answer.

**Part D — Production Guardrails**
You'll close with the safety controls a real SQL agent needs: read-only access, allowlisted schema, SQL parsing, row limits, logging, and observability.

---

## Critical Points

**The `base_url` pattern is the most important concept in this lab.** Memorize it:
```python
client = OpenAI(api_key=YOUR_KEY, base_url="https://api.openai.com/v1")
```
Every inference engine you encounter — FastAPI, vLLM, Ollama, Together AI, Groq — exposes an OpenAI-compatible endpoint. Change `base_url`, keep everything else. This is how teams swap backends without touching application code.

**`gpt-4o-mini` is your workhorse model.** It's cheap, fast, and capable enough for all labs. `gpt-4o` is used only for quality comparisons where you need to see the difference.

**HuggingFace `pipeline()` is a high-level convenience wrapper.** Under it lives `AutoTokenizer`, `AutoModel`, and a generation loop. Lab 3 takes you inside those layers.

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
| Tool/function calling | Letting the model request a structured function call that your application executes |
| ReAct | Reasoning and acting loop: think, call a tool, observe, answer |
| SQL agent | An agent that answers questions by generating and executing constrained SQL queries |
