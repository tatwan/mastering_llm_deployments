# Core Labs 0–7 Refinement — Design

**Date:** 2026-09-21
**Scope:** Labs 00–07 (`00_Environment_Check` … `07_Gradio_RAG_App`), their READMEs, root `README.md`, `.gitignore`, `.env.example`, `AGENTS.md`, `progress/`.
**Out of scope this pass:** Labs 08–12, Capstone A/B, Bonus. Same conventions apply to them in a later pass.

## Goal

Bring the core two-day labs in line with the instructor's teaching style:

1. Break complex topics into small, digestible chunks.
2. Prefer short cells (one-liners or a few lines) that let a student peek inside, then compile the pieces into a larger unit only once the pieces are understood.
3. Every notebook is self-guided: the surrounding text explains why, what to look at in the output, and what it means, for both Python experts and non-experts.
4. Notebooks run on Colab and locally without edits. Secrets come from Colab Secrets on Colab and from `.env` locally.
5. Simplicity wins. Good practice, but no defensive code or abstraction that does not serve the lesson.

## Conventions (every notebook in scope)

### Install cell

Three lines, identical shape everywhere, only the package list changes:

```python
import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} <packages>
```

`--python {sys.executable}` installs into the kernel's interpreter, which is Colab's system Python or the local venv. No `--system`, no `IN_COLAB` branch, no `shutil.which`. Lab 4 keeps its extra `torchao` upgrade line. Verified locally on 2026-09-21 with uv 0.12.5.

### Config cell (labs that call OpenAI: 1A, 1B, 2, 5, 6, 7)

```python
import os
try:
    from google.colab import userdata          # Colab: read the Secret
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except ImportError:
    from dotenv import load_dotenv             # local: read .env
    load_dotenv()
assert os.environ.get("OPENAI_API_KEY"), "Add OPENAI_API_KEY as a Colab Secret or to .env"

DEFAULT_MODEL = "gpt-4o-mini"
```

- The `get_secret()` helper is removed everywhere.
- Lab 5 loads `NGROK_AUTH_TOKEN` the same way (second `userdata.get` line, second assert).
- Optional keys (Groq in 1A and 5) use `os.environ.get("GROQ_API_KEY")` and skip the cell if absent.
- The instructor demonstrates adding a Colab Secret live, so notebooks carry no step-by-step Colab Secret instructions beyond the assert message. READMEs keep one sentence.
- Labs 0, 3, 4 need no key and get no config cell.
- `OPENAI_BASE_URL`, `DEFAULT_MODEL`, `QUALITY_MODEL`/`JUDGE_MODEL` constants stay where a lab uses them, defined next to first use.

### Repo files

- `.env.example` at repo root with `OPENAI_API_KEY=`, `NGROK_AUTH_TOKEN=`, `GROQ_API_KEY=` and one comment line each.
- `.gitignore` gains `.env`.
- Root `README.md` "Local Setup" gains: copy `.env.example` to `.env`, fill in keys, notebooks load it automatically.
- `python-dotenv` is already in `requirements.txt` and `pyproject.toml`; `uv` is added to both so local `%pip install -q uv` is a no-op.

### Text conventions

Follow Lab 3, which is the reference:

- A short "why" before each part.
- Where a cell is split, one or two sentences of markdown between the pieces saying what the next cell adds.
- A **Checkpoint** line after cells whose output matters: what you should see and what it means.
- No new markdown for its own sake. Labs 0–6 are close to the bar already; Lab 7 is thin and gets the fuller treatment.

## Per-lab changes

| Lab | Change |
|-----|--------|
| 0 | Install cell convention only. |
| 1A | Install + config conventions. Groq cell uses `os.environ.get`. |
| 1B | Install + config. After the memory agent is walked through step by step (existing cells 9–13), define one `run_agent(question, tools)` that takes a dict of tool name → function. Reuse it for the weather and SQL agents. The 59-line SQL cell becomes the tool schema plus one call. |
| 2 | Install + config. Split the 33-line cell into config and the `chat()` helper. |
| 3 | Install convention only. |
| 4 | Install convention (keep torchao line). Split the 35-line save cell and the 31-line reload cell where each does two things. No other edits; cannot execute without T4. |
| 5 | Install + config (adds `NGROK_AUTH_TOKEN`). Replace the 74-line `server.py` string with four `%%writefile` cells: (1) imports, app, backend config; (2) request models; (3) `/health` and `/v1/models`; (4) `/v1/chat/completions` with streaming. First cell uses `%%writefile server.py`, the rest `%%writefile -a server.py`, with a note to run top to bottom. Split the 52-line launch cell into "start uvicorn and wait for `/health`" and "open ngrok, fall back to localhost". |
| 6 | Install + config. Hybrid search becomes three cells: build BM25 and show a keyword-only result; define the RRF merge; compare semantic vs hybrid. Simplify the RRF lookup. RAGAS cell uses the current column names only (`user_input`, `response`, `retrieved_contexts`), one metrics import, one try/except with the rubric fallback. Verify the RAGAS API against current docs during implementation. |
| 7 | Install + config. Split the 43-line RAG cell into `retrieve()` and `rag_stream()`. Split the 60-line Gradio cell into callbacks (`respond`, `get_log`) and layout + wiring. Add Lab 3 style why/checkpoint text for each part. |

## Verification

- `ast.parse` every code cell in Labs 0–7 (skip `%%writefile` and `!`/`%` lines).
- Execute Labs 0 and 3 locally end to end with `jupyter nbconvert --execute`.
- Execute Labs 1A, 1B, 2, 6, 7 locally if `.env` holds a key at that point.
- Lab 4 (T4) and Lab 5 tunnel are not executed; Lab 5 server is smoke-tested locally with uvicorn if a key exists.
- Update each lab README where it describes install or secrets.
- Append a session entry to `progress/LOG.md`; refresh the install/secret lines in `AGENTS.md` and `progress/CURRENT_STATE.md`.

## Non-goals

No new labs, no renumbering, no LangChain rewrites, no dependency-pinning overhaul beyond adding `uv`. Do not touch Labs 8–12, Capstone, Bonus, slides.
