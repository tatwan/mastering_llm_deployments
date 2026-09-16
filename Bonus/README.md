# Bonus notebooks

Optional extras. They are **not** on the 2-day clock. Labs 8–12 (observability, cache, Docker, guardrails, golden-set eval) live at the repo root as the Production Readiness Pack.

Each notebook uses the same Colab install pattern as the core labs (`uv` + `--system`) and Colab Secrets for keys.

---

## Map

| # | File | After | Colab? | What is new vs the core labs |
|---|------|-------|--------|------------------------------|
| 01 | [Function calling + DuckDB](01_function_calling.ipynb) | Lab 1B | **READY** | Multi-tool agent over **CSV tables** (not the Lab 1B SQLite toy). Weather is one tool among three. |
| 02 | [RAG with LlamaIndex](02_rag_llamaindex.ipynb) | Lab 6 | **READY** | Same RAG idea, abstracted. Ships `sample_docs/` so Colab does not need PDFs. |
| 03 | [Text ReAct](03_react_agent.ipynb) | Lab 1B, Lab 2 | **READY** | Legacy **text** Thought/Action loop. Lab 1B already did JSON tools. No `eval()`. |
| 04 | [LiteLLM gateway](04_litellm_gateway.ipynb) | Lab 1A, Lab 5 | **READY** | Routing table + token/latency log. Optional Groq route. |
| 05 | [Hugging Face Spaces](05_hf_spaces_deployment.md) | Lab 7 | **Browser** (not Colab) | Persistent public URL for the Gradio RAG app. |
| 06 | [Ollama local `base_url`](06_ollama_local.md) | Lab 1A, Lab 5 | **NOT COLAB-COMPLETE** | Laptop: same OpenAI client → `http://localhost:11434/v1`. |

Colab badges:

- [01](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/01_function_calling.ipynb)
- [02](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/02_rag_llamaindex.ipynb)
- [03](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/03_react_agent.ipynb)
- [04](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/04_litellm_gateway.ipynb)

---

## When to open which

| If you want to… | Open |
|---|---|
| Drive tools from **files**, not a three-row SQLite table | 01 |
| See Lab 6’s pipeline as a library | 02 |
| Read a ReAct paper / debug regex agents | 03 |
| Route `fast` vs `quality` without rewriting the app | 04 |
| Keep a Gradio URL after class | 05 |
| Run the Lab 1A client with **no OpenAI key** on a laptop | 06 |

---

## What we did not add

- MCP server implementation (Bonus 03 names the standard only)
- vLLM install (Lab 5 concept; needs a GPU box)
- Kubernetes / SageMaker (out of course scope)

OpenAI key: 01–04 and Spaces. 06 uses a local model. 05 is a Space secret, not a notebook cell.
