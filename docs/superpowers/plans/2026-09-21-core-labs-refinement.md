# Core Labs 0–7 Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Labs 0–7 match the instructor's style: one short install cell, one short Colab-or-.env config cell, no big code blobs, self-guided text, same notebook runs on Colab and locally.

**Architecture:** Each lab is one `.ipynb` edited in place with a small JSON helper (cells are replaced or inserted by original index, applied high-to-low so indices stay valid). Shared repo files (`.env.example`, `.gitignore`, root README, requirements) change once. Every lab is syntax-checked; CPU labs are executed locally from a scratch copy with the install cell removed so the instructor's venv is not modified.

**Tech Stack:** Jupyter notebooks (nbformat 4.5, cells carry `id`), Python 3.12 venv at `.venv`, uv 0.12, python-dotenv, nbclient for execution.

**Spec:** `docs/superpowers/specs/2026-09-21-core-labs-refinement-design.md`

## Global Constraints

- Install cell is exactly three lines: `import sys` / `%pip install -q uv` / `!uv pip install -q --python {sys.executable} <packages>`. Lab 4 adds a fourth line for torchao.
- Config cell: `try: from google.colab import userdata` → `os.environ[...] = userdata.get(...)`; `except ImportError: load_dotenv()`; one `assert` per required key with message `"Add <NAME> as a Colab Secret or to .env"`. No `get_secret()` anywhere.
- Add `python-dotenv` to the install line of every lab that has a config cell (1A, 1B, 2, 5, 6, 7).
- Optional Groq key (1A, 5): a 5-line `try/except` in the Groq cell itself, never in the config cell.
- Labs 0, 3, 4 get no config cell.
- No new helper functions that exist only to hide code. No nested try/except.
- Do not touch Labs 8–12, Capstone, Bonus, `slides/`, `archive/`.
- Do not write notebook outputs back into the repo. Executed copies live in the scratchpad.
- Do not run the install cell against the instructor's `.venv` (it would downgrade gradio to 4.x). Execute scratch copies with that cell removed.
- Never commit `.env`. Commit after each lab.
- Scratchpad: `/private/tmp/claude-501/-Users-tarekatwan-Repos-MyWork-Teach-repos-master-llm-deployments/a6f6702f-30f6-44b5-9c8d-348ba5e5dca6/scratchpad` (referred to as `$SP` below).

---

### Task 1: Tooling — notebook edit helper, syntax check, executor

**Files:**
- Create: `$SP/nb.py`
- Create: `$SP/astcheck.py`
- Create: `$SP/run_nb.py`

**Interfaces:**
- Produces: `nb.apply(path, ops)` where `ops` is a list of `(index, action, payload)`; `action` ∈ `"replace"` (payload: str source), `"insert_after"` (payload: list of `(kind, source)`), `"delete"` (payload ignored). Indices are the ORIGINAL cell indices; the helper applies high index first.
- Produces: `python3 $SP/astcheck.py <nb>...` exits non-zero on a syntax error.
- Produces: `.venv/bin/python $SP/run_nb.py <nb> [--no-share]` executes a scratch copy (install cell removed) in the lab's folder, saves `$SP/executed/<name>.ipynb`, prints PASS/FAIL and the first error.

- [ ] **Step 1: Write the edit helper**

```python
# $SP/nb.py
import json, uuid

def load(path):
    return json.load(open(path))

def save(path, nb):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

def cell(kind, source):
    c = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {},
         "source": source.rstrip("\n").splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

def apply(path, ops):
    """ops: [(original_index, 'replace'|'insert_after'|'delete', payload)]."""
    nb = load(path)
    order = {"replace": 0, "insert_after": 1, "delete": 2}
    for idx, action, payload in sorted(ops, key=lambda o: (-o[0], order[o[1]])):
        if action == "replace":
            nb["cells"][idx]["source"] = payload.rstrip("\n").splitlines(keepends=True)
        elif action == "insert_after":
            nb["cells"][idx + 1:idx + 1] = [cell(k, s) for k, s in payload]
        elif action == "delete":
            del nb["cells"][idx]
    save(path, nb)
```

- [ ] **Step 2: Write the syntax checker**

```python
# $SP/astcheck.py
import ast, json, sys
bad = 0
for path in sys.argv[1:]:
    nb = json.load(open(path))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("%%writefile"):
            src = "\n".join(src.splitlines()[1:])          # check the file body as Python
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad += 1
            print(f"SYNTAX {path} cell {i}: {e}")
print("OK" if not bad else f"{bad} bad cells")
sys.exit(1 if bad else 0)
```

- [ ] **Step 3: Write the executor**

```python
# $SP/run_nb.py
import json, os, sys, pathlib, traceback
import nbformat
from nbclient import NotebookClient

src = pathlib.Path(sys.argv[1]).resolve()
no_share = "--no-share" in sys.argv
nb = nbformat.read(src, as_version=4)
nb.cells = [c for c in nb.cells if not (c.cell_type == "code" and "uv pip install" in c.source)]
if no_share:
    for c in nb.cells:
        if c.cell_type == "code":
            c.source = c.source.replace("share=True", "share=False")
out_dir = pathlib.Path(__file__).parent / "executed"; out_dir.mkdir(exist_ok=True)
client = NotebookClient(nb, timeout=900, kernel_name="python3",
                        resources={"metadata": {"path": str(src.parent)}})
try:
    client.execute()
    print("PASS", src.name)
except Exception as e:
    print("FAIL", src.name); print(str(e)[:3000])
finally:
    nbformat.write(nb, out_dir / src.name)
```

- [ ] **Step 4: Install executor deps into the venv and verify the checker on the current notebooks**

Run:
```bash
cd /Users/tarekatwan/Repos/MyWork/Teach/repos/master_llm_deployments
uv pip install -q --python .venv/bin/python nbformat nbclient
python3 $SP/astcheck.py 0[0-7]_*/*.ipynb
```
Expected: `OK` (baseline before any edit).

---

### Task 2: Repo files — `.env.example`, `.gitignore`, README, requirements

**Files:**
- Create: `.env.example`
- Modify: `.gitignore`
- Modify: `README.md` (lines 23–41: secrets paragraph and Local Setup)
- Modify: `requirements.txt` (add `uv`)
- Modify: `pyproject.toml` (add `uv` to dependencies)

- [ ] **Step 1: Create `.env.example`**

```
# Copy this file to .env in the same folder and paste your keys after the = sign.
# The notebooks read .env automatically when they run outside Colab. Never commit .env.

OPENAI_API_KEY=

# Lab 5 only. Free token from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN=

# Optional. Labs 1A and 5 add a Groq provider row if this is set.
GROQ_API_KEY=
```

- [ ] **Step 2: Add `.env` to `.gitignore`** — append under `# Tooling`:

```
.env
```

- [ ] **Step 3: Edit root `README.md`**

Replace the line `The notebooks load it automatically from Colab Secrets. Most notebooks also support a local `OPENAI_API_KEY` environment variable for instructor testing.` with:

```
The notebooks load it automatically from Colab Secrets. Running locally, they read the same names from a `.env` file instead (see Local Setup).
```

Replace the Local Setup block with:

````
### Local Setup

```bash
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
cp .env.example .env                  # then paste your keys into .env
jupyter lab
```

Each notebook's first cell installs its own packages with `uv` into whichever Python the kernel is using, so the same cell works in Colab and on your laptop. Keys come from Colab Secrets in Colab and from `.env` locally.
````

- [ ] **Step 4: Add `uv` to `requirements.txt`** under `# Notebook utilities`:

```
uv>=0.5.0
```

and to `pyproject.toml` dependencies (alphabetical, after `trl`):

```
    "uv>=0.5.0",
```

- [ ] **Step 5: Verify and commit**

Run: `git status --short` — expect `.env.example`, `.gitignore`, `README.md`, `requirements.txt`, `pyproject.toml` changed and `.env` NOT listed as untracked.

```bash
git add .env.example .gitignore README.md requirements.txt pyproject.toml
git commit -m "Add .env.example and local .env setup; document the uv install cell"
```

---

### Task 3: Lab 0 — install convention

**Files:**
- Modify: `00_Environment_Check/lab0_env_check.ipynb` cells 1 (md), 2 (code), 20 (md)
- Modify: `00_Environment_Check/README.md` line 36

- [ ] **Step 1: Apply edits**

```python
import sys; sys.path.insert(0, "$SP"); import nb
nb.apply("00_Environment_Check/lab0_env_check.ipynb", [
 (2, "replace", '''import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} transformers torch sentence-transformers chromadb openai langchain langchain-openai gradio'''),
 (1, "replace", '''---

## 1. Install the course packages

**Why:** Colab starts empty. Each lab installs what it needs, but Lab 0 installs the set you will see all week so a surprise missing import does not show up in Lab 1.

**What:** `uv` is a fast installer. `%pip install uv` gets it (Colab does not ship it; on a laptop that already has it, this is a no-op). The `--python {sys.executable}` flag tells uv to install into the Python this notebook is running, so the same line works in Colab and in your local environment.

**When:** Run this once per new Colab runtime. Locally, if you already installed from `requirements.txt`, you can skip it.'''),
])
```

Then in cell 20 (troubleshooting table) replace the row
`| `No virtual environment found` | `uv pip` on Colab without `--system` | The install cell already passes `--system` on Colab. Re-run it from the top. |`
with
`| `No virtual environment found` | `uv` could not see the notebook's Python | Re-run the install cell exactly as written; `--python {sys.executable}` points uv at this kernel. |`
(use `nb.load`, string-replace in cell 20's joined source, `nb.save`).

- [ ] **Step 2: README** — replace line 36 with:

```
**Local:** from the repo root, `uv pip install -r requirements.txt` (or `pip`), then open this notebook in Jupyter. The install cell also works locally; it installs into the kernel's Python.
```

- [ ] **Step 3: Check, execute, commit**

```bash
python3 $SP/astcheck.py 00_Environment_Check/lab0_env_check.ipynb
.venv/bin/python $SP/run_nb.py 00_Environment_Check/lab0_env_check.ipynb
```
Expected: `OK` then `PASS lab0_env_check.ipynb`.

```bash
git add 00_Environment_Check
git commit -m "Lab 0: three-line uv install cell that works on Colab and locally"
```

---

### Task 4: Lab 1A — install, config, Groq cell

**Files:**
- Modify: `01_Modern_Stack/lab1_modern_stack.ipynb` cells 1 (md), 2, 3, 27 (code)
- Modify: `01_Modern_Stack/README.md` line 8

**Interfaces:**
- Produces (used by later cells unchanged): `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `DEFAULT_MODEL`, `QUALITY_MODEL`.

- [ ] **Step 1: Apply edits**

```python
nb.apply("01_Modern_Stack/lab1_modern_stack.ipynb", [
 (27, "replace", '''try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
except Exception:                      # not on Colab, or no such secret
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

providers = {
    "OpenAI gpt-4o-mini": {"base_url": OPENAI_BASE_URL, "api_key": OPENAI_API_KEY, "model": DEFAULT_MODEL},
}
if GROQ_API_KEY:
    providers["Groq llama-3.1-8b-instant"] = {
        "base_url": "https://api.groq.com/openai/v1", "api_key": GROQ_API_KEY, "model": "llama-3.1-8b-instant",
    }
else:
    print("No GROQ_API_KEY — showing OpenAI only. A second provider is one more dict entry.\\n")

q = "In one sentence: what is PagedAttention?"
for name, cfg in providers.items():
    c = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    r = c.chat.completions.create(model=cfg["model"], messages=[{"role": "user", "content": q}])
    print(f"[{name}]\\n  {r.choices[0].message.content}\\n")'''),
 (3, "replace", '''import os
try:
    from google.colab import userdata          # Colab: read the Secret you added
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except ImportError:
    from dotenv import load_dotenv             # local: read .env in the repo root
    load_dotenv()
assert os.environ.get("OPENAI_API_KEY"), "Add OPENAI_API_KEY as a Colab Secret or to .env"

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL   = "gpt-4o-mini"   # workhorse for this course
QUALITY_MODEL   = "gpt-4o"        # quality / judge comparisons only
print(f"Ready — {DEFAULT_MODEL} at {OPENAI_BASE_URL}, key starts {OPENAI_API_KEY[:8]}...")'''),
 (2, "replace", '''import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} "transformers>=5" torch openai langchain langchain-openai langchain-core httpx python-dotenv'''),
 (1, "replace", '''---

## 0. Install and configure

**Why:** Colab starts empty. We need Transformers (local GPT-2), the OpenAI SDK (hosted calls), and LangChain (Part C).

**What:** The same three-line installer as Lab 0. Then a config cell that reads `OPENAI_API_KEY` from Colab Secrets when you are in Colab, or from the repo's `.env` file when you run locally. Either way the key lands in an environment variable and never in a cell.

**When:** Once per new runtime.'''),
])
```

- [ ] **Step 2: README** — line 8 becomes:

```
**Day 1 Morning | CPU | `OPENAI_API_KEY` required** (instructor provides; Colab Secret in Colab, `.env` locally)
```

- [ ] **Step 3: Check, execute if key present, commit**

```bash
python3 $SP/astcheck.py 01_Modern_Stack/lab1_modern_stack.ipynb
grep -q '^OPENAI_API_KEY=sk' .env && .venv/bin/python $SP/run_nb.py 01_Modern_Stack/lab1_modern_stack.ipynb || echo "no key: static only"
git add 01_Modern_Stack/lab1_modern_stack.ipynb 01_Modern_Stack/README.md
git commit -m "Lab 1A: short install and Colab-or-.env config cells"
```

---

### Task 5: Lab 1B — install, config, one `run_agent`

**Files:**
- Modify: `01_Modern_Stack/lab1_part2_tools_react_sql_agent.ipynb` cells 2, 3, 14 (md), 15, 18, 26, 30 (md), 31

**Interfaces:**
- Produces: `run_agent(question, tools, functions, system=...) -> str` where `tools` is a list of tool schemas and `functions` maps tool name → Python callable returning a JSON string. Used by cells 15, 18, 26, 31.
- Consumes: `client`, `DEFAULT_MODEL`, `json`, `memory_tool`, `estimate_model_memory`, `weather_tool`, `get_current_weather`, `sql_tool`, `run_sql` (all already defined in the notebook).

- [ ] **Step 1: Apply edits**

```python
nb.apply("01_Modern_Stack/lab1_part2_tools_react_sql_agent.ipynb", [
 (31, "replace", '''# TODO: implement get_efficiency_score, write its tool schema, then call run_agent with BOTH tools.

def get_efficiency_score(name: str) -> str:
    raise NotImplementedError("Query benchmarks.db and return JSON with an efficiency field")


# After you implement it:
# print(get_efficiency_score("Qwen2.5-1.5B"))
# print(run_agent("Which model has the best quality-per-speed efficiency?",
#                 tools=[sql_tool, efficiency_tool],
#                 functions={"run_sql": run_sql, "get_efficiency_score": get_efficiency_score}))'''),
 (30, "replace", '''**Checkpoint:** the 8 GB answer should name models from **your** table (Qwen2.5-0.5B / 1.5B / Llama-3.2-3B), not a random internet list.

---

## Exercise — add a second tool

Add `get_efficiency_score(name)` that looks up `quality` and `tokens_per_sec` for one model and returns

`round(quality / tokens_per_sec * 100, 2)`

as JSON. Then ask: **Which model has the best quality-per-speed efficiency?**

You will need a Python function, a tool schema, and one more entry in the `functions` dict you pass to `run_agent`.'''),
 (26, "replace", '''sql_tool = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": (
            "Run a read-only SQLite SELECT against the models table. "
            "Columns: name TEXT, params_b REAL, vram_gb REAL, "
            "tokens_per_sec INTEGER, quality REAL (0-10). "
            "SELECT only. No semicolons."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single read-only SELECT over models."}},
            "required": ["query"],
        },
    },
}

def sql_agent(question):
    return run_agent(question, tools=[sql_tool], functions={"run_sql": run_sql},
                     system="You answer questions about LLM benchmark data. "
                            "Use run_sql when the answer needs table data. Cite the SQL evidence.")

print("sql_agent ready")'''),
 (18, "replace", '''print(run_agent("What is the current weather in Amman, Jordan?",
                tools=[weather_tool], functions={"get_current_weather": get_current_weather}))'''),
 (15, "replace", '''def run_agent(question, tools, functions, system="You answer LLM deployment questions. Use tools when they help."):
    """One tool-calling round trip: ask → run whatever tools the model requested → ask again."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]

    first = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages, tools=tools)
    reply = first.choices[0].message
    if not reply.tool_calls:                    # the model answered directly
        return reply.content

    messages.append(reply)
    for call in reply.tool_calls:
        args = json.loads(call.function.arguments)
        print(f"Tool requested: {call.function.name}({args})")
        result = functions[call.function.name](**args)
        print("Observation   :", result[:180])
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    second = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages)
    return second.choices[0].message.content

memory_tools = {"estimate_model_memory": estimate_model_memory}
print(run_agent("How much weight memory does a 7B model need in INT4 versus FP16?", [memory_tool], memory_tools))
print()
print(run_agent("How much memory does a 7B model need in fp8?", [memory_tool], memory_tools))'''),
 (14, "replace", '''Those three roles — `user`, `assistant` (with `tool_calls`), `tool` — are the protocol. LangChain, LlamaIndex, and AutoGen all assemble this structure under the hood.

Now wrap the loop **once** as `run_agent(question, tools, functions)`:

- `tools` — the list of schemas the model is allowed to choose from
- `functions` — a dict from tool name to the Python function that actually runs it

Parts B and D reuse this one function. Only the tools change.'''),
 (3, "replace", '''import os
try:
    from google.colab import userdata          # Colab: read the Secret you added
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except ImportError:
    from dotenv import load_dotenv             # local: read .env in the repo root
    load_dotenv()
assert os.environ.get("OPENAI_API_KEY"), "Add OPENAI_API_KEY as a Colab Secret or to .env"

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL   = "gpt-4o-mini"

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
print(f"Ready — {DEFAULT_MODEL} at {OPENAI_BASE_URL}")'''),
 (2, "replace", '''import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} openai pandas httpx python-dotenv'''),
])
```

- [ ] **Step 2: Fix the weather checkpoint (cell 19)** — its first sentence stays true; append after "only the tool changed.": ` The model chose the tool on its own this time (`tool_choice` defaulted to `auto`).`

- [ ] **Step 3: Check, execute if key present, commit**

```bash
python3 $SP/astcheck.py 01_Modern_Stack/lab1_part2_tools_react_sql_agent.ipynb
grep -q '^OPENAI_API_KEY=sk' .env && .venv/bin/python $SP/run_nb.py 01_Modern_Stack/lab1_part2_tools_react_sql_agent.ipynb || echo "no key: static only"
git add 01_Modern_Stack/lab1_part2_tools_react_sql_agent.ipynb
git commit -m "Lab 1B: one run_agent reused for memory, weather, and SQL tools"
```

---

### Task 6: Lab 2 — install, config split from `chat()`

**Files:**
- Modify: `02_Prompting/lab2_prompting.ipynb` cells 1, 2; insert two cells after 2

- [ ] **Step 1: Apply edits**

```python
nb.apply("02_Prompting/lab2_prompting.ipynb", [
 (2, "replace", '''import os
try:
    from google.colab import userdata          # Colab: read the Secret you added
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except ImportError:
    from dotenv import load_dotenv             # local: read .env in the repo root
    load_dotenv()
assert os.environ.get("OPENAI_API_KEY"), "Add OPENAI_API_KEY as a Colab Secret or to .env"

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL   = "gpt-4o-mini"

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
print(f"Ready — {DEFAULT_MODEL} at {OPENAI_BASE_URL}")'''),
 (2, "insert_after", [
  ("markdown", '''### One helper: `chat()`

Every cell in this lab sends one prompt and reads one answer. `chat()` does that in five lines so the cells below show only the prompt. Pass `system=` to put instructions on the trust boundary (Part C explains why that matters).'''),
  ("code", '''def chat(prompt, model=DEFAULT_MODEL, temperature=0.7, system=None):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return response.choices[0].message.content

print(chat("Say hello in five words."))'''),
 ]),
 (1, "replace", '''import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} openai tiktoken python-dotenv'''),
])
```

- [ ] **Step 2: Check, execute if key present, commit**

```bash
python3 $SP/astcheck.py 02_Prompting/lab2_prompting.ipynb
grep -q '^OPENAI_API_KEY=sk' .env && .venv/bin/python $SP/run_nb.py 02_Prompting/lab2_prompting.ipynb || echo "no key: static only"
git add 02_Prompting/lab2_prompting.ipynb
git commit -m "Lab 2: config cell separated from the chat() helper"
```

---

### Task 7: Lab 3 — install convention

**Files:**
- Modify: `03_Inspect_Chat/lab3_inspect_chat.ipynb` cell 1

- [ ] **Step 1: Apply edit**

```python
nb.apply("03_Inspect_Chat/lab3_inspect_chat.ipynb", [
 (1, "replace", '''import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} transformers torch accelerate'''),
])
```

- [ ] **Step 2: Check, execute, commit** (Qwen2.5-0.5B ~1 GB download on first run; allow up to 15 min)

```bash
python3 $SP/astcheck.py 03_Inspect_Chat/lab3_inspect_chat.ipynb
.venv/bin/python $SP/run_nb.py 03_Inspect_Chat/lab3_inspect_chat.ipynb
git add 03_Inspect_Chat/lab3_inspect_chat.ipynb
git commit -m "Lab 3: three-line install cell"
```
Expected: `PASS lab3_inspect_chat.ipynb`.

---

### Task 8: Lab 4 — install, split save and reload cells

**Files:**
- Modify: `04_Quantize_LoRA/lab4_quantize_lora.ipynb` cells 1, 2 (md), 16, 17; insert after 16 and 17

**Interfaces:**
- Consumes: `model_peft`, `tokenizer`, `vram_nf4`, `MODEL_ID`, `bnb_config`, `torch`, `AutoModelForCausalLM` (defined earlier in the notebook).
- Produces: `adapter_path`, `base`, `chat_with`, `base_answers`, `tuned`, `tuned_chat` (cell 19 uses `base_answers` and `tuned_chat`).

- [ ] **Step 1: Apply edits**

```python
nb.apply("04_Quantize_LoRA/lab4_quantize_lora.ipynb", [
 (17, "replace", '''# Reload the NF4 base model fresh, plus a small chat helper we will use for both models
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map='auto')

def chat_with(model, q, max_new_tokens=150):
    msgs      = [{'role': 'user', 'content': q}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs    = tokenizer(formatted, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print(chat_with(base, 'What is QLoRA?'))'''),
 (17, "insert_after", [
  ("markdown", '''That was the **base** model, before any fine-tuning. Save its answers to three questions, then load the adapter on top of the same base object. `PeftModel.from_pretrained` does not copy the 1.5B weights; it attaches the ~10 MB adapter.'''),
  ("code", '''comparison_questions = ['What is QLoRA?', 'How does LoRA reduce trainable parameters?', 'What is NF4 quantization?']
base_answers = {q: chat_with(base, q) for q in comparison_questions}

tuned = PeftModel.from_pretrained(base, adapter_path)

def tuned_chat(q):
    return chat_with(tuned, q)

print('Q:', comparison_questions[0])
print('A:', tuned_chat(comparison_questions[0]))'''),
 ]),
 (16, "replace", '''import os

adapter_path = './my_lora_adapter'
model_peft.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

total = 0
for f in os.listdir(adapter_path):
    size = os.path.getsize(os.path.join(adapter_path, f))
    total += size
    print(f'  {f}: {size/1e6:.1f} MB')
print(f'Adapter package : {total/1e6:.1f} MB')
print(f'Base model      : ~{vram_nf4*1000:.0f} MB')'''),
 (16, "insert_after", [
  ("markdown", '''**Checkpoint:** the adapter folder is a few MB; the base model is thousands. You ship the adapter. Users download the base once, and one base can serve many adapters.

### A model card travels with the adapter

Hugging Face Hub and most registries read a `README.md` with YAML front matter. Writing it now costs six lines and makes `push_to_hub` a one-liner later.'''),
  ("code", '''model_card = f"""---
base_model: {MODEL_ID}
library_name: peft
tags: [qlora, fine-tuned, llm-deployment]
---
# My QLoRA adapter
Base `{MODEL_ID}` · adapter {total/1e6:.1f} MB · QLoRA (NF4 + LoRA r=16) · 10 examples × 3 epochs
"""
open(f'{adapter_path}/README.md', 'w').write(model_card)
print(open(f'{adapter_path}/README.md').read())'''),
 ]),
 (2, "replace", '''`torchao` is a PyTorch quantization helper some Colab images ship in an old version; the extra upgrade line avoids a known bitsandbytes/transformers clash. If the GPU check below fails, you selected CPU — stop and switch to T4 before loading weights.'''),
 (1, "replace", '''import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} transformers torch accelerate bitsandbytes peft "trl>=0.16" datasets
!uv pip install -q --python {sys.executable} --upgrade torchao'''),
])
```

- [ ] **Step 2: Check the before/after cell still only uses `base_answers`, `tuned_chat`, `comparison_questions`** — it is the code cell whose source starts with `print("--- BEFORE / AFTER ---")` (old index 19; indices shift by +4 after the inserts, so find it by content). Read it. If it references anything else from the old cell 17 (e.g. `base_chat`), add a one-line alias in the new cell: `base_chat = lambda q: chat_with(base, q)`.

- [ ] **Step 3: Check and commit** (no execution: T4 only)

```bash
python3 $SP/astcheck.py 04_Quantize_LoRA/lab4_quantize_lora.ipynb
git add 04_Quantize_LoRA/lab4_quantize_lora.ipynb
git commit -m "Lab 4: split adapter save/model card and base/adapter reload into single-purpose cells"
```

---

### Task 9: Lab 5 — install, config, server in four cells, launch in two

**Files:**
- Modify: `05_Serving_API/lab5_serving_api.ipynb` cells 2, 3 (md), 4, 6 (md), 8, 9 (md), 10, 19; insert after 8 and after 10

**Interfaces:**
- Produces: `server.py` (file), `LOCAL_URL`, `SERVER_URL`, `NGROK_HEADERS`, `server_proc`, `ngrok` (used by cells 13, 15, 23 unchanged).
- Consumes: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `DEFAULT_MODEL`, `NGROK_AUTH_TOKEN` from the config cell.

- [ ] **Step 1: Apply edits**

```python
nb.apply("05_Serving_API/lab5_serving_api.ipynb", [
 (19, "replace", '''try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
except Exception:                      # not on Colab, or no such secret
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

providers = {
    "Our FastAPI → OpenAI":        {"base_url": f"{SERVER_URL}/v1", "api_key": "not-needed", "model": DEFAULT_MODEL},
    "OpenAI direct (gpt-4o-mini)": {"base_url": OPENAI_BASE_URL, "api_key": OPENAI_API_KEY, "model": DEFAULT_MODEL},
}
if GROQ_API_KEY:
    providers["Groq llama-3.1-8b-instant"] = {
        "base_url": "https://api.groq.com/openai/v1", "api_key": GROQ_API_KEY, "model": "llama-3.1-8b-instant"}
else:
    print("No GROQ_API_KEY — comparing your server vs OpenAI only.\\n")

question = "In one sentence: what is PagedAttention?"
for name, cfg in providers.items():
    c = make_client(cfg["base_url"], cfg["api_key"])
    r = c.chat.completions.create(model=cfg["model"], messages=[{"role": "user", "content": question}])
    print(f"[{name}]\\n  {r.choices[0].message.content}\\n")'''),
 (10, "replace", '''import subprocess, time, httpx, sys

LOCAL_URL = "http://127.0.0.1:8000"
env = {**os.environ, "BACKEND_API_KEY": OPENAI_API_KEY, "BACKEND_BASE_URL": OPENAI_BASE_URL, "DEFAULT_MODEL": DEFAULT_MODEL}

# Logs go to a file. Never PIPE stdout on Colab: a full pipe freezes uvicorn.
log = open("uvicorn.log", "w")
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000"],
    stdout=log, stderr=subprocess.STDOUT, env=env,
)

healthy = False
for _ in range(20):                       # up to 10 seconds
    try:
        httpx.get(f"{LOCAL_URL}/health", timeout=1).raise_for_status()
        healthy = True
        break
    except Exception:
        time.sleep(0.5)

assert healthy, "Server did not start. Open uvicorn.log for the reason."
print("Local health OK —", LOCAL_URL)'''),
 (10, "insert_after", [
  ("markdown", '''**Checkpoint:** `Local health OK`. The server is alive inside this runtime, but nothing outside can reach port 8000 yet.

### Step A3 — Open the public tunnel

ngrok gives the local port an HTTPS URL. If the tunnel fails (bad token, network policy), the lab continues on localhost — the OpenAI-compatible server is the lesson, the public URL is the demo.'''),
  ("code", '''from pyngrok import ngrok

ngrok.set_auth_token(NGROK_AUTH_TOKEN)
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}   # free plan shows a browser warning page; API calls skip it

try:
    SERVER_URL = ngrok.connect(8000).public_url
    print("Public URL :", SERVER_URL)
    print("Swagger UI :", f"{SERVER_URL}/docs")
    print("If a browser shows an ngrok warning page, click Visit Site.")
except Exception as e:
    SERVER_URL = LOCAL_URL
    print("ngrok tunnel did not open:", e)
    print("Continuing on", SERVER_URL)'''),
 ]),
 (9, "replace", '''### Step A2 — Launch the server

This cell starts uvicorn in the background on `127.0.0.1:8000` and polls `GET /health` until it answers. Logs go to `uvicorn.log` (never a stdout pipe — a full pipe freezes the server on Colab). If health never comes up, open `uvicorn.log`.

Step A3 then opens an ngrok tunnel with `pyngrok` (still [ngrok's official Colab path](https://ngrok.com/docs/using-ngrok-with/googleColab)). You need a free account and the `NGROK_AUTH_TOKEN` secret. Gradio (Lab 7) uses its own `share=True` tunnel instead.

**Colab / free-plan quirks (read once):**

| What you will see | What to do |
|---|---|
| Browser interstitial “Visit Site” on the ngrok URL | Click **Visit Site**. That warning is for *browsers*. Our Python client sends `ngrok-skip-browser-warning` so API calls skip it. |
| `ERR_NGROK_107` / invalid authtoken | Check the `NGROK_AUTH_TOKEN` secret. Copy the token from the ngrok dashboard again. |
| Health check on localhost fails | Port 8000 did not come up. Open `uvicorn.log`, or run the cleanup cell and retry A2. |
| Tunnel fails but localhost health is OK | Keep going: B1–B3 use `http://127.0.0.1:8000`. You still built an OpenAI-compatible server. |
| Port 8000 already in use | Cleanup cell, then A2 again. |

After a successful tunnel: open **Swagger** (`/docs`) on your phone. That is the “this is a real API” moment.'''),
 (8, "replace", '''%%writefile server.py
import os, json, time, uuid
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="My LLM API", version="0.1.0")

# Which backend this server forwards to. Change the env vars and restart; nothing else changes.
BACKEND_API_KEY  = os.environ.get("BACKEND_API_KEY", "")
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL    = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")

backend = OpenAI(api_key=BACKEND_API_KEY, base_url=BACKEND_BASE_URL)'''),
 (8, "insert_after", [
  ("markdown", '''### A1.2 — Request schemas

Pydantic models define the JSON the server accepts. If a caller sends the wrong shape, FastAPI returns `422` before your function runs. These two classes *are* the OpenAI request format. `-a` appends to the file.'''),
  ("code", '''%%writefile -a server.py

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = 500'''),
  ("markdown", '''### A1.3 — Two small routes

`/health` is what a load balancer polls. `/v1/models` is what an OpenAI client can list. Each is a plain function with a decorator.'''),
  ("code", '''%%writefile -a server.py

@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND_BASE_URL, "model": DEFAULT_MODEL}

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": DEFAULT_MODEL, "object": "model", "owned_by": "custom-server"}]}'''),
  ("markdown", '''### A1.4 — The main route

`POST /v1/chat/completions` has two paths. **Non-streaming:** forward to the backend, then reshape the reply into the OpenAI response format. **Streaming:** `generate()` yields one Server-Sent-Events line per token delta (`data: {...}\\n\\n`) and finishes with `data: [DONE]`. That is why ChatGPT looks like it is typing.'''),
  ("code", '''%%writefile -a server.py

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    if not req.stream:
        resp = backend.chat.completions.create(
            model=req.model, messages=msgs, temperature=req.temperature, max_tokens=req.max_tokens
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": resp.choices[0].message.content}}],
            "usage": resp.usage.model_dump(),
        }

    def generate():
        stream = backend.chat.completions.create(
            model=req.model, messages=msgs, temperature=req.temperature, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                data = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                yield f"data: {json.dumps(data)}\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(generate(), media_type="text/event-stream")'''),
  ("markdown", '''**Checkpoint:** four cells, one file. Print it to see the whole server in one place — this is what production would copy into a container.'''),
  ("code", '''print(open("server.py").read())'''),
 ]),
 (6, "replace", '''### How to Read the Server Cells

`server.py` is written in **four short cells**. The first creates the file (`%%writefile server.py`); the next three append to it (`%%writefile -a`). Run them top to bottom once. If you edit one, re-run all four from A1.1.

| Cell | What it adds | Why it matters |
|---|---|---|
| A1.1 | app object + backend config from env vars | Swap the backend by changing env vars, not code |
| A1.2 | Pydantic request schemas | Wrong JSON is rejected before your code runs |
| A1.3 | `/health`, `/v1/models` | What load balancers and clients probe |
| A1.4 | `/v1/chat/completions` with streaming | The OpenAI wire format, including SSE |

> **The deployment lesson:** today the server forwards to OpenAI. In production, point `BACKEND_BASE_URL` at vLLM and nothing else changes.'''),
 (4, "replace", '''import os
try:
    from google.colab import userdata          # Colab: read the Secrets you added
    os.environ["OPENAI_API_KEY"]   = userdata.get("OPENAI_API_KEY")
    os.environ["NGROK_AUTH_TOKEN"] = userdata.get("NGROK_AUTH_TOKEN")
except ImportError:
    from dotenv import load_dotenv             # local: read .env in the repo root
    load_dotenv()
assert os.environ.get("OPENAI_API_KEY"),   "Add OPENAI_API_KEY as a Colab Secret or to .env"
assert os.environ.get("NGROK_AUTH_TOKEN"), "Add NGROK_AUTH_TOKEN as a Colab Secret or to .env"

OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
NGROK_AUTH_TOKEN = os.environ["NGROK_AUTH_TOKEN"]
OPENAI_BASE_URL  = "https://api.openai.com/v1"
DEFAULT_MODEL    = "gpt-4o-mini"
print(f"Ready — {DEFAULT_MODEL}, ngrok token found")'''),
 (3, "replace", '''### 📦 What Was Just Installed

- **fastapi** — the web framework. You define routes and schemas in Python; FastAPI turns them into an HTTP API with automatic validation and documentation.
- **uvicorn** — the ASGI server. FastAPI is just a Python object; uvicorn is the process that binds to a port, listens for HTTP requests, and calls your FastAPI code.
- **pyngrok** — the Python client for ngrok. It opens a public tunnel from a URL on ngrok's servers to a local port inside this runtime.
- **openai** — the OpenAI Python SDK. We use it both to *call* OpenAI (from our server) and to *call our own server* (as a client), proving they are interchangeable.
- **httpx** — an HTTP client. Used for health checks.'''),
 (2, "replace", '''import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} fastapi uvicorn pyngrok openai httpx python-dotenv'''),
])
```

- [ ] **Step 2: Rename the section heading for A1.1** — cell 7 (markdown "Part A — Write and Launch the Server") ends with `> **Instructor note:** Read this top to bottom. Four endpoints. One file.` Replace that line with `### A1.1 — App object and backend config`.

- [ ] **Step 3: Local smoke test of the server file** (needs a key in `.env`)

```bash
python3 $SP/astcheck.py 05_Serving_API/lab5_serving_api.ipynb
grep -q '^OPENAI_API_KEY=sk' .env && grep -q '^NGROK_AUTH_TOKEN=.' .env && .venv/bin/python $SP/run_nb.py 05_Serving_API/lab5_serving_api.ipynb || echo "static only"
```
If `NGROK_AUTH_TOKEN` is empty locally, the assert fires by design; then test the server half by hand: extract the four `%%writefile` bodies to `$SP/server.py`, run `BACKEND_API_KEY=$(grep OPENAI_API_KEY .env | cut -d= -f2) .venv/bin/python -m uvicorn server:app --port 8000` from `$SP`, and `curl localhost:8000/health`. Expected `{"status":"ok",...}`. Kill uvicorn after.

- [ ] **Step 4: Commit**

```bash
git add 05_Serving_API/lab5_serving_api.ipynb
git commit -m "Lab 5: server.py built in four short cells; launch and tunnel split; .env config"
```

---

### Task 10: Lab 6 — install, config, hybrid search in three cells, RAGAS without nesting

**Files:**
- Modify: `06_RAG_Pipeline/lab6_rag_pipeline.ipynb` cells 2, 3, 29, 32; insert after 29 and 32

**Interfaces:**
- Produces: `keyword_search(query, n=3) -> list[str]`, `hybrid_search(query, n=3, k=60) -> list[str]`.
- Consumes: `all_chunks`, `collection`, `search()`, `rag()`, `JUDGE_MODEL`.

- [ ] **Step 1: Apply edits**

```python
nb.apply("06_RAG_Pipeline/lab6_rag_pipeline.ipynb", [
 (32, "replace", '''# RAGAS: an LLM judge (gpt-4o) scores each answer on two metrics.
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))

rows = []
for q in ["What is NF4 quantization?", "What is QLoRA?", "What is PagedAttention?"]:
    docs, _, _ = search(q)
    answer, _  = rag(q)
    rows.append({"user_input": q, "response": answer, "retrieved_contexts": list(docs)})

results = evaluate(EvaluationDataset.from_list(rows), metrics=[Faithfulness(), AnswerRelevancy()], llm=judge)
results.to_pandas()[["user_input", "faithfulness", "answer_relevancy"]]'''),
 (32, "insert_after", [
  ("markdown", '''**Checkpoint:** three rows, two scores each, most above 0.8. A low **faithfulness** means the generator added claims the retrieved text does not support. A low **answer relevancy** means the answer wandered off the question.

If the cell above raised (package API moved, judge quota), read the error, then do the same review by hand below. The habit is the lesson: inspect the answer, inspect the sources, ask whether every claim is supported.'''),
  ("code", '''# Manual rubric — always works, no judge model
for q in ["What is NF4 quantization?", "What is QLoRA?", "What is PagedAttention?"]:
    answer, sources = rag(q)
    print("Q:", q)
    print("A:", answer[:200], "...")
    print("Sources:", [s["source"] for s in sources])
    print("Grounded? Relevant?  (you decide)\\n")'''),
 ]),
 (29, "replace", '''import bm25s

bm25 = bm25s.BM25(corpus=all_chunks)
bm25.index(bm25s.tokenize(all_chunks))

def keyword_search(query, n=3):
    hits, _ = bm25.retrieve(bm25s.tokenize([query]), k=n)
    return hits[0].tolist()                       # chunk texts, best first

for text in keyword_search("NF4 bitsandbytes double quantization"):
    print("-", text[:100], "...")'''),
 (29, "insert_after", [
  ("markdown", '''BM25 matched the literal words. Now merge the two ranked lists with **Reciprocal Rank Fusion**: each list gives every chunk a vote of `1 / (k + rank)`, and we add the votes. A chunk that is near the top of *both* lists wins.'''),
  ("code", '''def hybrid_search(query, n=3, k=60):
    dense  = collection.query(query_texts=[query], n_results=n * 2)["documents"][0]
    sparse = keyword_search(query, n=n * 2)

    votes = {}
    for ranked in (dense, sparse):
        for rank, text in enumerate(ranked):
            votes[text] = votes.get(text, 0) + 1 / (k + rank + 1)

    return sorted(votes, key=votes.get, reverse=True)[:n]

hybrid_search("NF4 bitsandbytes double quantization")[0][:120]'''),
  ("markdown", '''Compare on two kinds of query: one keyword-heavy (hybrid should win), one a paraphrase (both should do fine).'''),
  ("code", '''for q in ["NF4 bitsandbytes double quantization", "how do I make a model understand my documents"]:
    print("Query   :", q)
    print("Semantic:", search(q)[0][0][:110], "...")
    print("Hybrid  :", hybrid_search(q)[0][:110], "...")
    print()'''),
 ]),
 (3, "replace", '''import os
try:
    from google.colab import userdata          # Colab: read the Secret you added
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except ImportError:
    from dotenv import load_dotenv             # local: read .env in the repo root
    load_dotenv()
assert os.environ.get("OPENAI_API_KEY"), "Add OPENAI_API_KEY as a Colab Secret or to .env"

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL   = "gpt-4o-mini"        # generator
JUDGE_MODEL     = "gpt-4o"             # RAGAS judge (Part E)
EMBED_MODEL     = "all-MiniLM-L6-v2"   # local embeddings, no key
print(f"Ready — generate with {DEFAULT_MODEL}, judge with {JUDGE_MODEL}, embed with {EMBED_MODEL}")'''),
 (2, "replace", '''import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} sentence-transformers chromadb langchain langchain-community langchain-openai langchain-text-splitters openai ragas datasets scikit-learn matplotlib pypdf bm25s python-dotenv'''),
])
```

- [ ] **Step 2: Update the Lab 6 Complete checklist** — the markdown cell containing `## ✅ Lab 6 Complete` (old index 33; find by content, indices shifted by +6). — replace `- [ ] RAGAS scores or manual rubric output` with `- [ ] Hybrid vs semantic comparison printed` and `- [ ] RAGAS table (or the manual rubric if RAGAS errored)`. Delete stretch goal 3 (hybrid search is now in Part D) and renumber 4 → 3.

- [ ] **Step 3: Check, execute if key present, commit**

```bash
python3 $SP/astcheck.py 06_RAG_Pipeline/lab6_rag_pipeline.ipynb
grep -q '^OPENAI_API_KEY=sk' .env && .venv/bin/python $SP/run_nb.py 06_RAG_Pipeline/lab6_rag_pipeline.ipynb || echo "no key: static only"
```
Note: the local venv's `ragas 0.4.3` currently fails to import (`langchain_community.chat_models.vertexai` missing). If the RAGAS cell FAILs locally for that reason and every earlier cell passed, record it as an environment issue in the log, not a notebook bug. Colab installs fresh.

```bash
git add 06_RAG_Pipeline/lab6_rag_pipeline.ipynb
git commit -m "Lab 6: hybrid search in three cells, RAGAS cell without nesting, .env config"
```

---

### Task 11: Lab 7 — install, config, split RAG and Gradio cells, self-guided text

**Files:**
- Modify: `07_Gradio_RAG_App/lab7_gradio_rag_app.ipynb` cells 1, 2, 4, 7, 9, 10; insert after 4, 7, 9, 10

**Interfaces:**
- Produces: `retrieve(query, n=3) -> (docs, metas)`, `rag_stream(question)` generator yielding `(answer_so_far, sources_md, latency_ms|None)`, `respond`, `get_log`, `demo`, `query_log`.

- [ ] **Step 1: Apply edits**

```python
nb.apply("07_Gradio_RAG_App/lab7_gradio_rag_app.ipynb", [
 (10, "replace", '''# share=True mints a public https://*.gradio.live URL (temporary, like the Lab 5 ngrok tunnel).
demo.launch(share=True, debug=False, quiet=True)'''),
 (10, "insert_after", [
  ("markdown", '''**Checkpoint:** two URLs printed — a local one and a `*.gradio.live` one. Open the public one on your phone. Ask one of the example questions and watch three things: tokens arrive one at a time, the Sources panel fills in, and the Query Log gains a row with a latency number.'''),
 ]),
 (9, "replace", '''query_log = []                       # in-memory log for this session

def respond(message, chat_history):
    """Gradio calls this on every message. It yields, so the UI updates per token."""
    if not message.strip():
        yield "", chat_history, ""
        return
    chat_history = chat_history + [(message, "")]
    latency = None
    for answer_so_far, sources_so_far, latency in rag_stream(message):
        chat_history[-1] = (message, answer_so_far)
        yield "", chat_history, sources_so_far
    query_log.append({"time": datetime.now().strftime("%H:%M:%S"), "query": message[:60], "latency_ms": latency})

def get_log():
    if not query_log:
        return "No queries yet."
    rows = [f"| {e['time']} | {e['query']} | {e['latency_ms']} |" for e in query_log[-10:]]
    return "| Time | Query | ms |\\n|---|---|---|\\n" + "\\n".join(rows)

print("Callbacks ready")'''),
 (9, "insert_after", [
  ("markdown", '''`respond` returns three values on every yield, in this order: the textbox (cleared), the chat history, the sources panel. The layout below wires those three outputs to three components. Keep the order in your head; it is the only contract between the backend and the UI.

### C2 — Layout and wiring'''),
  ("code", '''import gradio as gr

with gr.Blocks(title="LLM Course Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 LLM Deployment Course Assistant\\nAsk about **quantization, RAG, LoRA, serving, or fine-tuning**.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=420, show_copy_button=True)
            with gr.Row():
                msg      = gr.Textbox(placeholder="Ask about quantization, RAG, LoRA, vLLM...", show_label=False, scale=5, container=False)
                send_btn = gr.Button("Send ▶", variant="primary", scale=1)
            gr.Examples(inputs=msg, examples=[
                "What memory savings does NF4 give vs FP16?",
                "When should I use RAG vs fine-tuning?",
                "What makes vLLM faster than a naive FastAPI server?",
            ])
        with gr.Column(scale=2):
            sources_box = gr.Markdown("*Sources appear here after your first question.*")
            with gr.Accordion("Query Log", open=False):
                log_display = gr.Markdown("No queries yet.")
                refresh_btn = gr.Button("Refresh", size="sm")

    send_btn.click(respond, [msg, chatbot], [msg, chatbot, sources_box])
    msg.submit(respond,     [msg, chatbot], [msg, chatbot, sources_box])
    refresh_btn.click(get_log, outputs=log_display)

print("App built. Launch it in the next cell.")'''),
 ]),
 (7, "replace", '''from openai import OpenAI
import time
from datetime import datetime

oai = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

RAG_PROMPT = """You are an expert assistant for an LLM deployment course.
Answer ONLY based on the context below. Be concise and cite the source topic.
If the context does not cover the question, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

def retrieve(query, n=3):
    res = collection.query(query_texts=[query], n_results=n)
    return res["documents"][0], res["metadatas"][0]

docs, metas = retrieve("What is NF4 quantization?")
for d, m in zip(docs, metas):
    print(f"[{m['source']}] {d[:80]}...")'''),
 (7, "insert_after", [
  ("markdown", '''**Checkpoint:** three chunks, each tagged with its topic. That is the same retrieval as Lab 6, minus the distances.

### B2 — Stream the answer

`rag_stream` is a **generator**: instead of returning once, it `yield`s the partial answer every time a token arrives. Gradio consumes a generator the same way a `for` loop does, and redraws the chat each time. The sources panel is built before the first token so the user sees *where* the answer will come from while it is still typing.'''),
  ("code", '''def rag_stream(question):
    """Yields (answer_so_far, sources_markdown, latency_ms). latency_ms is None until the last yield."""
    t0 = time.time()
    docs, metas = retrieve(question)
    context    = "\\n\\n".join(f"[{m['source']}]: {d}" for d, m in zip(docs, metas))
    sources_md = "**📚 Retrieved Sources**\\n" + "".join(f"\\n**{i+1}. {m['source']}**\\n_{d[:100]}..._\\n" for i, (d, m) in enumerate(zip(docs, metas)))

    answer = ""
    stream = oai.chat.completions.create(model=DEFAULT_MODEL, stream=True,
                                         messages=[{"role": "user", "content": RAG_PROMPT.format(context=context, question=question)}])
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            answer += delta
            yield answer, sources_md, None

    latency_ms = int((time.time() - t0) * 1000)
    yield answer, sources_md + f"\\n\\n_⏱ {latency_ms} ms_", latency_ms

for answer, sources, latency in rag_stream("What is NF4 quantization?"):
    pass                                    # drain the generator like Gradio will
print(f"{latency} ms —", answer[:120], "...")'''),
  ("markdown", '''**Checkpoint:** a latency in milliseconds and the first line of a grounded answer. If you see a full answer here, the UI will stream it.'''),
 ]),
 (4, "replace", '''# A small inline knowledge base keeps the app self-contained. Lab 6 showed how to load PDFs and web pages instead.
knowledge_base = {
    "quantization": "Quantization reduces weight precision. NF4 achieves ~4x memory reduction vs FP16 with minimal quality loss. Double quantization saves another 0.4 bits/param. AWQ protects activation-salient weights. GGUF is the CPU format used by Ollama and llama.cpp for local deployment.",
    "rag":          "RAG retrieves documents at inference time and injects them into the prompt. Four stages: Load, Chunk, Embed, Retrieve+Generate. Hybrid search combines semantic and BM25. RAGAS evaluates faithfulness and answer relevancy.",
    "lora":         "LoRA adds trainable rank-r matrices BA to frozen weights. QLoRA combines NF4 base with 16-bit LoRA adapters. Rank 16 is a good starting point. Adapters are 10-100 MB, saved separately from the base model.",
    "serving":      "vLLM uses PagedAttention for non-contiguous KV-cache pages — up to ~24x throughput vs naive serving. Continuous batching serves N concurrent users on one GPU. SGLang uses RadixAttention to share KV prefixes. All expose OpenAI-compatible endpoints.",
    "finetuning":   "Fine-tuning changes model weights permanently. Use it for stable, repeated behaviors: tone, format, domain terminology. RAG is better for changing facts. Full fine-tuning stores a new giant per task; LoRA/QLoRA adapters are tiny (10-100 MB) and swap at runtime.",
}
print(len(knowledge_base), "topics")'''),
 (4, "insert_after", [
  ("markdown", '''Chunk, embed, and index — the Lab 6 pipeline in one short cell. The collection is in memory (no `./chroma_db` folder) because the app only needs it while it runs.'''),
  ("code", '''import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
chunks   = splitter.split_documents([Document(page_content=v, metadata={"source": k}) for k, v in knowledge_base.items()])

collection = chromadb.Client().get_or_create_collection(
    "lab7_kb", embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL))
collection.add(documents=[c.page_content for c in chunks],
               metadatas=[c.metadata for c in chunks],
               ids=[f"c{i}" for i in range(len(chunks))])
print(f"Knowledge base ready: {collection.count()} chunks")'''),
 ]),
 (2, "replace", '''import os
try:
    from google.colab import userdata          # Colab: read the Secret you added
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except ImportError:
    from dotenv import load_dotenv             # local: read .env in the repo root
    load_dotenv()
assert os.environ.get("OPENAI_API_KEY"), "Add OPENAI_API_KEY as a Colab Secret or to .env"

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL   = "gpt-4o-mini"
EMBED_MODEL     = "all-MiniLM-L6-v2"
print(f"Ready — {DEFAULT_MODEL}, embeddings {EMBED_MODEL}")'''),
 (1, "replace", '''import sys
%pip install -q uv
!uv pip install -q --python {sys.executable} "gradio>=4.44.1,<5" sentence-transformers chromadb langchain langchain-community langchain-text-splitters openai python-dotenv'''),
])
```

Note: the old install line had an unquoted `gradio>=4.44.1,<5`, which the shell reads as a redirect. The quotes fix that.

- [ ] **Step 2: Rename the Part B heading** — the markdown cell containing `## Part B — Streaming RAG Core` (find by content); replace its body with:

```
---

## Part B — Streaming RAG Core (15 min)

Two pieces: `retrieve` (same as Lab 6) and `rag_stream` (new: a generator). Build and test each on its own before Gradio touches them.

### B1 — Retrieve
```

and the markdown cell containing `## Part C — Gradio App` (find by content): append `\n\n### C1 — Callbacks\n\nTwo plain functions. `respond` streams one answer; `get_log` renders the log. Neither knows Gradio exists yet.` after its existing paragraph.

- [ ] **Step 3: Delete the old `chroma.get_or_create_collection` rerun-guard** — not needed: the new cell uses a fresh in-memory `chromadb.Client()` each run, so `collection.add` never collides. (Already handled by the replacement above; verify no cell references `chroma` or `ef`.)

```bash
python3 - <<'EOF'
import json; nb=json.load(open("07_Gradio_RAG_App/lab7_gradio_rag_app.ipynb"))
for i,c in enumerate(nb["cells"]):
    s="".join(c["source"])
    if c["cell_type"]=="code" and (" ef" in s or "chroma." in s): print("check cell", i)
EOF
```
Expected: no output.

- [ ] **Step 4: Check, execute (no public share), commit**

```bash
python3 $SP/astcheck.py 07_Gradio_RAG_App/lab7_gradio_rag_app.ipynb
grep -q '^OPENAI_API_KEY=sk' .env && .venv/bin/python $SP/run_nb.py 07_Gradio_RAG_App/lab7_gradio_rag_app.ipynb --no-share || echo "no key: static only"
```
Local venv has gradio 6, so the `gr.Chatbot` tuple format may warn or error locally. If only the launch/Blocks cells fail with a gradio-6-specific message, record it and rely on the Colab pin `<5`. If `rag_stream` or earlier fails, fix it.

```bash
git add 07_Gradio_RAG_App/lab7_gradio_rag_app.ipynb
git commit -m "Lab 7: retrieve/stream and callbacks/layout in separate cells with checkpoints"
```

---

### Task 12: Final verification and records

**Files:**
- Modify: `progress/LOG.md` (prepend entry), `progress/CURRENT_STATE.md` (tech stack "Package install in class" row + keys section), `AGENTS.md` (Conventions: per-notebook installs and secrets), `progress/AUDIT.md` (only if it lists per-lab cell counts — update the 00–07 rows).

- [ ] **Step 1: Re-run the stats and the syntax check across all of 0–7**

```bash
python3 $SP/astcheck.py 0[0-7]_*/*.ipynb
python3 $SP/nb_stats.py | head -12
grep -l "get_secret\|IN_COLAB\|--system" 0[0-7]_*/*.ipynb || echo "no leftovers"
```
Expected: `OK`; no lab 0–7 has a code cell over 45 lines; `no leftovers`.

- [ ] **Step 2: Prepend to `progress/LOG.md`** (newest at top, after the `---` under the header):

```
## 2026-09-21 — Labs 0–7 refined to the instructor's style (small cells, .env, uv --python)

**Who:** Agent session (Claude). Request: review and refine labs to match the teaching style — small digestible cells, self-guided text, Colab-or-local without edits, simplicity over engineering.

**Conventions now in 0–7:** three-line install (`%pip install uv` + `uv pip install --python {sys.executable}`); 8-line config (Colab Secrets → `.env` via python-dotenv → assert); no `get_secret()`; `.env.example` at root; `.env` gitignored.

**Splits:** Lab 5 `server.py` in four `%%writefile` cells, launch and tunnel separate; Lab 7 retrieve/stream and callbacks/layout separate; Lab 6 BM25 → RRF → compare, RAGAS without nesting plus a manual-rubric cell; Lab 1B one `run_agent` reused by memory, weather, SQL; Lab 2 config vs `chat()`; Lab 4 save/model-card and base/adapter reload.

**Executed locally:** <fill from run_nb results: PASS/FAIL per lab>. Lab 4 (T4) and the ngrok half of Lab 5 not executed.

**Not touched:** Labs 8–12, Capstone, Bonus. Same conventions apply there in a later pass.

**Spec / plan:** `docs/superpowers/specs/2026-09-21-core-labs-refinement-design.md`, `docs/superpowers/plans/2026-09-21-core-labs-refinement.md`.
```

- [ ] **Step 3: Update `AGENTS.md` Conventions**

Replace the `**Secrets.**` bullet with:
```
- **Secrets.** Config cell: `from google.colab import userdata` → `os.environ`, else `load_dotenv()` from the repo-root `.env`; one `assert` per required key. `OPENAI_API_KEY` (Labs 1A–2, 5–7), `NGROK_AUTH_TOKEN` (Lab 5), optional `GROQ_API_KEY` (1A, 5). `.env.example` is committed; `.env` is gitignored. No `get_secret()` helper.
```
Replace the `**Per-notebook installs**` bullet with:
```
- **Per-notebook installs** (Labs 0–7 as of 2026-09-21): three lines — `import sys` / `%pip install -q uv` / `!uv pip install -q --python {sys.executable} <pkgs>`. Works on Colab and in a local venv; no `--system`, no `IN_COLAB` branch. Labs 8–12 and Bonus still use the older three-branch block until their pass. Lab 10 writes Docker files only.
```
Add under **Known remaining work**: `8. Labs 8–12, Capstone, Bonus: apply the 2026-09-21 install/config conventions and cell-size pass.`

- [ ] **Step 4: Update `progress/CURRENT_STATE.md`** — in the Tech stack table, the "Package install in class" row becomes: `Labs 0–7: `%pip install uv` then `uv pip install --python {sys.executable}` (same line on Colab and locally). Labs 8–12, Bonus: older Colab/uv/pip branch.` In "Keys, accounts, and network", replace `Never paste keys into cells. `get_secret()` raises with Colab Secret instructions.` with `Never paste keys into cells. Labs 0–7 read Colab Secrets or `.env` (see `.env.example`); an assert names the missing key.`

- [ ] **Step 5: Commit the tracked pieces and report**

```bash
git status --short
git add docs/superpowers/plans/2026-09-21-core-labs-refinement.md
git commit -m "Add implementation plan for core labs refinement"
```
(`progress/` and `AGENTS.md` are gitignored; they are updated on disk only.)

Report to the instructor: which labs PASSed execution, which were static-only and why, the Lab 6 ragas local-import note, the Lab 7 gradio-6 local note, and that Labs 8–12/Capstone/Bonus are the next pass.
