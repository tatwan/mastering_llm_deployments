# Lab 5 — Serving Models as an API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/05_Serving_API/lab5_serving_api.ipynb)

**Day 2 Morning | ~45 minutes | CPU | OpenAI API key + ngrok account required**

---

## Purpose

Running a model locally in a notebook is a proof of concept. Serving it as an API is what makes it useful to anyone else. This lab closes that gap: you'll build an OpenAI-compatible HTTP server, expose it to the internet via ngrok, and call it from the same client code you used in Lab 1. Then you'll learn why FastAPI isn't the end of the story — and what vLLM does differently.

---

## What You Will Build

**Part A — FastAPI OpenAI-Compatible Server**
You'll write a server with three endpoints:
- `GET /health` — liveness check
- `GET /v1/models` — model listing (required by the OpenAI SDK)
- `POST /v1/chat/completions` — the main endpoint, supporting both synchronous and streaming (SSE) responses

The server proxies requests to an OpenAI-compatible backend, which means in development it calls `gpt-4o-mini`, but in production you'd swap the backend to vLLM or any other engine — without changing the server code.

**Part B — Expose and Test**
Launch the server inside Colab with `uvicorn`, then create a public URL with `pyngrok`. You'll need a free ngrok account — sign up at [ngrok.com](https://ngrok.com) and add your authtoken as a Colab Secret named `NGROK_AUTH_TOKEN`. Test it with `httpx` (raw HTTP) and then with the `OpenAI` SDK pointing at your server's URL. Run a provider comparison: your server, OpenAI direct, and OpenAI `gpt-4o`.

**Part C — Understanding vLLM**
A conceptual deep dive into what happens at scale. You'll see why naive single-request serving collapses under concurrent load, and how PagedAttention and continuous batching solve the two core bottlenecks.

---

## Critical Points

**The OpenAI-compatible endpoint standard is the most important interface in LLM serving.** Every major inference engine — vLLM, Ollama, Together AI, Groq, Mistral, Fireworks — exposes `POST /v1/chat/completions`. One client, swappable backends. Your application code never changes; only `base_url` does.

**Streaming uses Server-Sent Events (SSE).** Instead of waiting for the full response, the server sends tokens as they're generated using `text/event-stream`. Each chunk is a partial JSON delta. The client reassembles them. This is why ChatGPT "types" — the tokens arrive one at a time, not all at once.

**FastAPI is the right choice when:**
- You need custom business logic (auth, logging, routing, pre/post-processing)
- You're serving multiple models or mixing model types
- You want full control over the request/response lifecycle

**FastAPI is not the right choice when:**
- You need maximum throughput for a single LLM at scale
- You have many concurrent users
- You need automatic batching and KV-cache optimization

For high-throughput LLM serving, **vLLM** is the industry standard.

**Why KV-cache matters.** Attention is quadratic: every new token must attend to all previous tokens. The key-value matrices for previous tokens can be cached to avoid recomputation. vLLM's **PagedAttention** manages this cache like virtual memory in an OS — efficiently allocating and sharing it across requests. Without this, memory fragmentation causes 60–80% of GPU memory to be wasted.

**Continuous batching vs static batching:**
- Static: wait to fill a batch, process all at once, return all at once. Simple but slow.
- Continuous: new requests join the batch as soon as a slot opens. Keeps the GPU fully utilized.

At scale, the difference is 20–30× throughput.

---

## Key Terms

| Term | Definition |
|------|-----------|
| OpenAI-compatible endpoint | Any HTTP server implementing the `/v1/chat/completions` schema |
| SSE | Server-Sent Events — HTTP streaming where the server pushes chunks to the client |
| ngrok | Tunnel service that gives a public HTTPS URL to a local server |
| KV-cache | Cache of key-value attention matrices for previously processed tokens |
| PagedAttention | vLLM's memory management system for the KV-cache, inspired by OS paging |
| Continuous batching | Processing requests as they arrive rather than waiting to fill a fixed batch |
| Throughput | Requests (or tokens) processed per second across all concurrent users |
| Latency | Time-to-first-token for a single request |
