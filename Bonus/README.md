# Bonus notebooks

Optional extras, **not** on the 2-day clock. Each one picks up where a core lab leaves off, and they are numbered in the order you will reach those labs. Labs 8–12 at the repo root are the Production Readiness Pack; these are the side roads.

Every notebook installs its own packages in its first cell, the same way the core labs do, and reads keys from Colab Secrets on Colab or from the repo's `.env` locally.

---

## At a glance

| # | Bonus | After | Time | Runs on | Open |
|---|---|---|---|---|---|
| 01 | Tools over real tables (DuckDB) | Lab 1B | ~20 min | Colab CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/01_function_calling.ipynb) |
| 02 | Text ReAct against a real model | Lab 1B | ~15 min | Colab CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/02_react_agent.ipynb) |
| 03 | Serve your own model with vLLM | Labs 4, 5 | ~45 min | **Colab T4 GPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/03_vllm_serving.ipynb) |
| 04 | LiteLLM: one gateway in front of every model | Lab 5 | ~30 min | Colab CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/04_litellm_gateway.ipynb) |
| 05 | Ollama: a model on your own laptop | Lab 5 | ~20 min | Your laptop | [Read the guide](05_ollama_local.md) |
| 06 | The Lab 6 pipeline as a library (LlamaIndex) | Lab 6 | ~20 min | Colab CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/06_rag_llamaindex.ipynb) |
| 07 | Deploy your app to Hugging Face Spaces | Lab 7 | ~30 min | Browser | [Read the guide](07_hf_spaces_deployment.md) |

Times are rough and do not include first-time downloads. Bonus 03's install alone takes three to five minutes.

---

## What each one adds

**01 · Tools over real tables.** Lab 1B's agent read a three-row SQLite table. Here it gets three tools over CSV files loaded into DuckDB, and calls two of them in the same turn. Watch for the moment the model adds a currency the table never mentioned.

**02 · Text ReAct against a real model.** Lab 1B Part C broke the regex-parsed agent loop with replies written by hand. This one lets a real model break it: loosen the system prompt and the loop quietly returns prose as the answer, with no error anywhere.

**03 · Serve your own model with vLLM.** The serving engine Lab 5 Part C described, running on a free Colab T4. You serve the Qwen model from Lab 4 with your adapter merged in, call it with the usual OpenAI client, read how vLLM splits GPU memory between weights and KV cache, and send 32 requests at once to measure continuous batching.

**04 · LiteLLM gateway.** Apps ask for `"fast"` or `"quality"`; a routing table decides what that means. You see the price of every call, watch a fallback rescue a request sent to a broken provider, and then run the gateway as its own server, the way most teams deploy it.

**05 · Ollama.** The same client from Lab 1A pointed at a model on your laptop, with no API key and nothing leaving the machine. Optionally, put Lab 5's server in front of it.

**06 · LlamaIndex.** Lab 6's pipeline in about five lines, using the same free MiniLM embeddings and the same model, so the only difference is the abstraction. Then you open it up: the sources it retrieved and the grounding prompt it wrote for you.

**07 · Hugging Face Spaces.** Lab 7's `gradio.live` link dies with your Colab runtime. A Space is a small git repo that stays online, using the template in `hf_spaces_template/`.

---

## Before you start Bonus 03

Bonus 03 serves the adapter **you** trained in Lab 4, and that adapter disappears when the Lab 4 runtime does. Before you close Lab 4, run this in its notebook, then download the zip from the Files panel:

```
!zip -r my_lora_adapter.zip my_lora_adapter
```

No adapter? Bonus 03 still runs; it serves the base model instead. If you pushed your adapter to the Hugging Face Hub (Lab 4 stretch goal 4), Bonus 03 can pull it from there.

Select the **T4 GPU** runtime before running anything: Runtime → Change runtime type → T4 GPU.

---

## The serving trio: 03, 04, 05

The course's one idea is "same client, swap `base_url`". These three are backends you can swap to, and they fit together:

```
your app ── OpenAI client ──► LiteLLM gateway (04) ──┬──► OpenAI / Groq
                                                     ├──► vLLM on a GPU (03)
                                                     └──► Ollama on a laptop (05)
```

- **vLLM (03)** serves a model on a GPU for many users at once. It is where your Lab 4 fine-tune finally gets served.
- **Ollama (05)** runs a model on your own laptop, for a demo or when nothing may leave the machine.
- **LiteLLM (04)** sits in front of all of them, so your app never has to name a provider.

---

## Which one to open

| If you want to… | Open |
|---|---|
| Give an agent tools over real data files | 01 |
| See why text-parsed agents fail silently | 02 |
| Serve the model you fine-tuned in Lab 4 | 03 |
| Route between models, track cost, survive an outage | 04 |
| Run a model with no API key at all | 05 |
| Decide whether a RAG framework is worth it | 06 |
| Keep your Gradio app online after class | 07 |

## Accounts and keys

| Bonus | Needs |
|---|---|
| 01, 02, 06 | `OPENAI_API_KEY` |
| 04 | `OPENAI_API_KEY`; `GROQ_API_KEY` optional, adds a third route |
| 03 | No key. A T4 runtime. A free Hugging Face account only if you pull your adapter from the Hub. |
| 05 | No key. Ollama installed on your laptop. |
| 07 | A free Hugging Face account, and `OPENAI_API_KEY` as a Space secret (never in a file) |

## What these do not cover

- **Building an MCP server.** Bonus 02 explains what MCP is and where it fits.
- **Serving a LoRA adapter directly with vLLM on a T4.** On that GPU it is unreliable, so Bonus 03 merges the adapter first and explains why. It also gives the direct command for newer GPUs (L4, A100), where it works.
- **Kubernetes or SageMaker.** Out of scope for this course.

## Note for instructors

Bonus 03 was written against the vLLM documentation of September 2026 but has not yet been run end to end on a Colab T4. Run it once before recommending it; its troubleshooting table covers the most likely failures. Everything else here was executed on 2026-09-23 (Bonus 05 on macOS, Bonus 07's template smoke-tested but not deployed).
