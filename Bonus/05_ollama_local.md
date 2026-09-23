# Bonus 05 — Point the Same Client at Ollama

**Optional | After Lab 5 | Your laptop | No API key**

Checked 2026-09-23 with Ollama 0.33 on macOS: every step below, including step 4.

**Colab:** **not complete.** Ollama needs a local (or remote) daemon. This is the laptop counterpart to Lab 10.

The course architecture is “one OpenAI client, swap `base_url`.” You have pointed that client at OpenAI, optional Groq, and your Lab 5 FastAPI proxy. **Ollama** is the missing local backend: a process on your machine that speaks `/v1/chat/completions`.

---

## What you will do

1. Install Ollama and pull a small instruct model
2. Confirm `http://localhost:11434/v1` is alive
3. Copy the Lab 1A client, change **two strings**
4. (Optional) Point the Lab 5 FastAPI `BACKEND_BASE_URL` at Ollama instead of OpenAI

---

## Step 1 — Install and pull

https://ollama.com — install for your OS, then:

```bash
ollama pull qwen2.5:0.5b
ollama serve          # if it is not already running
```

`qwen2.5:0.5b` matches the family you inspected in Lab 3. A larger tag (`qwen2.5:1.5b`, `llama3.2`) works if your RAM allows.

## Step 2 — Health

```bash
curl -s http://localhost:11434/v1/models
```

You should see a JSON list. If this fails, Ollama is not listening — start `ollama serve`.

## Step 3 — Same client as Lab 1A

```python
from openai import OpenAI

client = OpenAI(
    api_key="ollama",                      # required by the SDK, ignored by Ollama
    base_url="http://localhost:11434/v1",
)

print(
    client.chat.completions.create(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": "What is QLoRA in one sentence?"}],
    )
    .choices[0]
    .message.content
)
```

That is the whole lesson. No new SDK.

Read the answer before you trust it. When we ran this, the 0.5B model explained QLoRA as a variant of "the Lora model" for reinforcement learning. It is fluent and wrong, the same warning Lab 3 gave about small models. Running locally changes where the model lives, not how much it knows.

## Step 4 — Optional: Lab 5 proxy in front of Ollama

On the machine where Ollama runs (not Colab), in a folder containing Lab 5's `server.py` (the Lab 5 cell that prints the file shows all of it):

```bash
export BACKEND_API_KEY=ollama
export BACKEND_BASE_URL=http://localhost:11434/v1
export DEFAULT_MODEL=qwen2.5:0.5b
uvicorn server:app --port 8000
```

Apps still call *your* FastAPI `base_url`. You swapped the backend by changing env vars — Lab 5’s config story.

## When to use this vs vLLM vs OpenAI

| Backend | Use when |
|---------|----------|
| OpenAI / Groq | Class, no GPU, quality and tools |
| Ollama | Laptop demo, air-gapped tryout, Lab 3-sized models |
| vLLM | A GPU and real concurrency (Lab 5 Part C, [Bonus 03](03_vllm_serving.ipynb)) |
| Your FastAPI proxy | Custom logic in front of any of the above |

## Bonus 05 complete

- [ ] `curl` listed a model
- [ ] Lab 1A-shaped client printed a local completion
- [ ] You can explain why `api_key="ollama"` is a dummy

This is not Colab. Do not spend 2-day clock time installing Ollama in the room unless everyone is on laptops.

Next: [Bonus 06 — RAG with LlamaIndex](06_rag_llamaindex.ipynb), or put a gateway in front of Ollama and OpenAI together with [Bonus 04](04_litellm_gateway.ipynb) (`ollama/qwen2.5:0.5b` is a LiteLLM route).
