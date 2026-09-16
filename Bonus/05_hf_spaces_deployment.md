# Bonus 05 — Deploy the Gradio RAG App to Hugging Face Spaces

**Optional | After Lab 7 | Browser | OpenAI API key as a Space secret**

**Colab:** not needed. This is a browser + Git (or the Hugging Face web UI) lab.

Lab 7’s `share=True` URL dies with the Colab runtime (hours, not days). A Hugging Face Space is a **git repo that stays up**. No Docker, no AWS, no ngrok.

Docs: [Gradio Spaces](https://huggingface.co/docs/hub/main/spaces-sdks-gradio) · [Space dependencies](https://huggingface.co/docs/hub/spaces-dependencies)

---

## What you will deploy

A simplified Lab 7 RAG assistant (inline course KB, MiniLM, Chroma, `gpt-4o-mini`):

```text
Bonus/hf_spaces_template/
  app.py
  requirements.txt
  README.md          ← Space metadata (sdk: gradio)
```

No LangChain. Four short documents are stored whole. That is enough to prove retrieve → generate → cite.

---

## Step 1 — Create the Space

1. https://huggingface.co/spaces → **Create new Space**
2. SDK: **Gradio** · Hardware: **CPU basic** · Visibility: your choice
3. Create

## Step 2 — Secret

Settings → **Repository secrets** → add `OPENAI_API_KEY`. Never put the key in `app.py`.

## Step 3 — Upload the three files

Put them at the **root** of the Space (web UI or `git push`). The Space rebuilds.

| Symptom | Likely cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not set` | Missing secret | Add it, then **Factory reboot** |
| Import error | `requirements.txt` incomplete | Match the template |
| Answers fail | Bad key | Replace the secret |
| Slow first question | Cold start + MiniLM download | Wait and retry |

## Step 4 — Test

1. In-scope: “What is QLoRA?”
2. Serving: “Why does vLLM help throughput?”
3. Out-of-scope: “What is the weather today?”

The first two should cite sources. The third should decline.

## What this is not

No auth, rate limits, traces (Lab 8), or golden-set gate (Lab 12). It is a **persistent public URL** — the missing piece after Lab 7.

Next: [Bonus 06 — Ollama](06_ollama_local.md) if you want a local `base_url`, or the [Capstone](../Capstone/README.md).
