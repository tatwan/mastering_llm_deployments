# Lab 5 — Serving Models as an API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/05_Serving_API/lab5_serving_api.ipynb)

**Day 2 Morning | ~45 minutes | CPU | OpenAI API key + ngrok account required**

**Colab status:** ready, with a localhost fallback. Official ngrok docs still use `pyngrok` on Colab. If the tunnel fails, the OpenAI-compatible server still runs on `127.0.0.1:8000`.

---

## Coming from Lab 4

Lab 4 produced a LoRA adapter on a GPU. This notebook does **not** serve that adapter — Colab CPU cannot run vLLM. You will write an OpenAI-compatible **FastAPI proxy** in front of `gpt-4o-mini`, then point the Lab 1A client at *your* ngrok URL. Same `base_url` swap; now you own the server.

Secrets: `OPENAI_API_KEY` and `NGROK_AUTH_TOKEN` (free at [ngrok.com](https://ngrok.com)). Optional: `GROQ_API_KEY`.

---

## Purpose

Since Lab 1A you have been the client. Here you write the server behind the URL: an OpenAI-compatible FastAPI app, about fifty lines, that forwards to `gpt-4o-mini` and answers in the OpenAI format. Then you call it with the same client code you already know, and read what changes once a real model sits behind it.

---

## What You Will Build

**Part A — Write and launch the server.** `server.py` in five short cells, with three routes:
- `GET /health` — is the process alive
- `GET /v1/models` — what the OpenAI SDK lists
- `POST /v1/chat/completions` — the main route, plain or streaming (SSE)

uvicorn runs it in the background on `127.0.0.1:8000`, logging to `uvicorn.log` (not a stdout pipe, which can freeze the server on Colab). `/health` is proved on localhost first, then `pyngrok` opens a public URL (ngrok's documented Colab path). Free ngrok shows a "Visit Site" page in a **browser**; the notebook's API calls send `ngrok-skip-browser-warning` and skip it. If the tunnel fails, everything continues on localhost.

**Part B — Call it.** A health check and a deliberately bad request (FastAPI returns `422` before your code runs). The stock OpenAI client pointed at your URL. Streaming through the SDK, then the raw `data:` lines with plain `httpx`. One loop over several backends: your server, OpenAI, and Groq (`openai/gpt-oss-20b`) if a `GROQ_API_KEY` secret exists.

**Part C — What your server cannot do.** Reading, no code. Put a model on your own GPU behind this server and users queue, the KV cache runs out of memory, and 4-bit weights run slowly. Continuous batching, PagedAttention and quantized kernels are vLLM's answers, and vLLM exposes the same route you just wrote. [Bonus 03](../Bonus/03_vllm_serving.ipynb) runs vLLM on a Colab T4 and measures the batching gain.

---

## Critical Points

**The OpenAI format is the interface, not the vendor.** vLLM, Ollama, TGI, Groq, Together, Fireworks and LiteLLM all expose `POST /v1/chat/completions`. Your client code outlives any one backend. Only `base_url` changes.

**Streaming is Server-Sent Events.** The connection stays open and the server writes one `data: {...}` line per chunk, ending with `data: [DONE]`. Nothing about it is specific to LLMs.

**The proxy is not the bottleneck.** FastAPI runs plain `def` handlers in a thread pool, so a server that forwards to OpenAI copes with a classroom of callers. The trouble starts when the handler runs the model itself: `generate()` serves one batch at a time, and a batch waits for its longest answer.

**The KV cache decides how many users fit.** Lab 3 measured it growing with every token of every conversation. Naive servers reserve the maximum length per request; the vLLM paper found existing systems wasting 60 to 80 % of KV cache memory that way. PagedAttention allocates the cache in small blocks as it grows and brings the waste under 4 %.

**Continuous batching vs static batching:**
- Static: collect a batch, run it, return it. Every request waits for the slowest one.
- Continuous: after every decoding step, finished sequences leave and waiting ones join. The GPU stays busy.

How much it adds up to depends on the model and the traffic. When vLLM launched in 2023, its authors measured up to 24× the throughput of plain Hugging Face Transformers and 2 to 4× over the best serving systems of the time.

**FastAPI and vLLM are not rivals.** FastAPI is where your logic goes: auth, logging, routing, guardrails. vLLM is where the model runs. Production often runs both, FastAPI in front and vLLM behind, which is this lab's shape with OpenAI swapped out.

---

## Key Terms

| Term | Definition |
|------|-----------|
| OpenAI-compatible endpoint | Any HTTP server implementing the `/v1/chat/completions` request and response shapes |
| SSE | Server-Sent Events — an open HTTP response the server writes `data:` lines into |
| ngrok | Tunnel service that gives a local port a public HTTPS URL |
| KV cache | Stored keys and values for tokens already processed, so each new token does not recompute them (Lab 3) |
| PagedAttention | vLLM's block-based allocation for the KV cache, modelled on OS memory paging |
| Continuous batching | Requests join and leave the running batch at every decoding step |
| Throughput | Tokens (or requests) per second across all users |
| Latency | How long one user waits. Time to first token matters for streaming UIs; total time matters for everything else (Lab 7 shows total) |

---

## Next

[Lab 6 — RAG Pipeline](../06_RAG_Pipeline/README.md) — retrieve course text with MiniLM and Chroma, generate with `gpt-4o-mini`. The generator is still an OpenAI-compatible call, so it could go through the server you built here.
