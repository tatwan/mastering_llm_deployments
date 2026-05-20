# Bonus 06 — Deploy the Gradio RAG App to Hugging Face Spaces

**Optional | After Lab 7 | Browser-based | OpenAI API key required as a Space secret**

Lab 7 gives you a temporary public Gradio URL from Colab. Hugging Face Spaces gives you a persistent public URL backed by a Git repository. This is the cleanest "ship it" extension for the course without asking students to learn Docker, AWS, Kubernetes, or local setup.

Official Hugging Face docs describe Gradio Spaces as Git repositories, initialized with `sdk: gradio` in the Space `README.md`, and the runtime installs dependencies from a root `requirements.txt`.

- Gradio Spaces docs: https://huggingface.co/docs/hub/main/spaces-sdks-gradio
- Spaces dependency docs: https://huggingface.co/docs/hub/spaces-dependencies

## What You Will Deploy

You will deploy a simplified version of the Lab 7 RAG assistant:

- `app.py` — the Gradio application
- `requirements.txt` — Python dependencies
- `README.md` — Space metadata with `sdk: gradio`
- Space secret: `OPENAI_API_KEY`

The template files are in:

```text
Bonus/hf_spaces_template/
```

## Step 1 — Create the Space

1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Choose:
   - SDK: **Gradio**
   - Hardware: **CPU basic** is enough for this template
   - Visibility: public or private
4. Create the Space.

## Step 2 — Add the Secret

In your Space:

1. Open **Settings**
2. Find **Repository secrets**
3. Add:
   - Name: `OPENAI_API_KEY`
   - Value: your classroom or instructor-provided key

Never paste API keys into `app.py`, `README.md`, or commit history.

## Step 3 — Upload the Template Files

Upload the three files from `Bonus/hf_spaces_template/` into the root of the Space repository:

```text
README.md
app.py
requirements.txt
```

You can use the Hugging Face web UI or Git. The web UI is easier for class; Git is better for production teams.

## Step 4 — Watch the Build

After upload, the Space will rebuild. Open the **Logs** tab if it fails.

Common failures:

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `OPENAI_API_KEY is not set` | Missing Space secret | Add the secret in Settings |
| Import error | Missing dependency | Add it to `requirements.txt` |
| App starts but answers fail | Bad or expired API key | Replace the secret |
| Slow first response | Cold start | Wait and retry |

## Step 5 — Test Deployment Behavior

Ask three questions:

1. A question in the knowledge base: "What is QLoRA?"
2. A serving question: "Why does vLLM help throughput?"
3. An out-of-scope question: "What is the weather today?"

The app should cite sources for the first two and decline the third.

## Reflection

This is not "production" yet. It is a public app deployment. Production adds authentication, persistent logs, rate limits, observability, data governance, and a cost model.

Still, this step matters: students leave with a real URL, not just a notebook output.

