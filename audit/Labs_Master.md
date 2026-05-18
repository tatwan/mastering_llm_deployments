# Mastering LLM Deployment
## Lab Guide — Claude Consolidated Edition
**Last Updated: May 2026 | For Coding Agent Implementation**

---

> **For the Coding Agent:** This document is the authoritative specification for all hands-on labs in the "Mastering LLM Deployment" 2-day intensive course. Each lab is a self-contained **Google Colab notebook**. Students need only a Google account — no local installation, no personal API keys, no GPU. The instructor provides a single temporary API key (Groq) that is shared with the class and deleted after the session.
>
> **Three Hard Constraints:**
> 1. **Cloud-native:** Every lab runs in Google Colab (free tier). No local install required.
> 2. **No personal API keys:** All LLM inference uses a temporary instructor-provided Groq key placed at the top of each notebook. Embeddings use `sentence-transformers` (runs locally inside Colab, no key needed).
> 3. **GPU is a bonus, not a requirement:** Labs 1, 2, 4, 5, 6 run on Colab CPU. Lab 3 (quantization/fine-tuning) benefits from the free T4 GPU — instructions included for switching runtime.
>
> **Implementation Principles:**
> - Every notebook opens with a `# 🔑 CONFIGURATION` cell where the student pastes the shared API key
> - All `pip install` cells are at the top, run once
> - Default LLM: Groq `llama-3.1-8b-instant` (fast, free, OpenAI-compatible)
> - Default embeddings: `all-MiniLM-L6-v2` from sentence-transformers (46MB, runs on Colab CPU in seconds)
> - Every lab ends with a working artifact students can screenshot or share
> - Include `# INSTRUCTOR NOTE:` comments in code cells for facilitation cues

---

## LAB MAP — Timing, Slot, and Runtime

```
DAY 1 MORNING              DAY 1 AFTERNOON            DAY 2 MORNING              DAY 2 AFTERNOON
────────────────────────   ────────────────────────   ────────────────────────   ────────────────────────
Lab 1: Stack Setup         Lab 3: Quantize + LoRA     Lab 4: Serve It            Lab 6: Deploy + UI
Lab 2: Load & Explore      [T4 GPU runtime]           Lab 5: Build RAG           Capstone

~45 min each               ~75 min                    ~45 min each               ~90 min capstone

Runtime: CPU               Runtime: T4 GPU            Runtime: CPU               Runtime: CPU
```

**Total lab time:** ~6 hours across 2 days
**Lecture:Lab ratio:** approximately 60:40

---

## API BACKEND USED IN THESE LABS

All LLM calls go to **Groq** — a free, fast, OpenAI-compatible inference service.

- **Why Groq?** Free tier, no credit card required, 14K tokens/min on 8B models, OpenAI-compatible (same client code)
- **Base URL:** `https://api.groq.com/openai/v1`
- **Models used:** `llama-3.1-8b-instant` (fast, demos), `llama-3.3-70b-versatile` (quality)
- **Key lifecycle:** Instructor creates key at class start → shares with students → deletes after Day 2

### Instructor: How to Create a Groq Key
```
1. Go to https://console.groq.com
2. Sign in (free account)
3. API Keys → Create API Key → name it "class-YYYYMMDD"
4. Share the key via the class chat/Slack/email at the start of Day 1
5. At end of Day 2: API Keys → Delete key
```

### Alternative Backends (same code, swap base_url + api_key)
```python
# Groq (default — fastest free option)
base_url="https://api.groq.com/openai/v1"
model="llama-3.1-8b-instant"

# Together AI (also free tier, 200+ models)
base_url="https://api.together.xyz/v1"
model="meta-llama/Llama-3.2-3B-Instruct-Turbo"

# HuggingFace Inference API (free tier)
base_url="https://api-inference.huggingface.co/v1/"
model="meta-llama/Llama-3.2-3B-Instruct"

# OpenAI (if students have personal keys or instructor upgrades)
base_url="https://api.openai.com/v1"
model="gpt-4o-mini"
```

---

## PRE-CLASS STUDENT INSTRUCTIONS

Send this to students the day before class (2–3 lines):

> Before class, go to [this Colab link] and run the first cell to verify your environment. You'll need a Google account (gmail.com). No other software or accounts are needed — we'll provide everything else on Day 1.

### `00_environment_check.ipynb` (pre-class verification notebook)

```python
# Cell 1 — Run this before class to confirm your environment is ready
# Takes about 60 seconds

print("Checking Python version...")
import sys
print(f"  Python: {sys.version}")

print("Installing packages...")
!pip install -q transformers sentence-transformers chromadb openai langchain langchain-community gradio

print("Testing core imports...")
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import gradio as gr

print("Testing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer("Hello, class!")
print(f"  Tokenizer works — token count: {len(tokens['input_ids'])}")

print("Testing embeddings (small model, ~60 sec first time)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
vec = embed_model.encode("test sentence")
print(f"  Embeddings work — dimension: {len(vec)}")

print("\n✅ Environment check PASSED — you're ready for class!")
print("Note: On Day 1 the instructor will share an API key for LLM inference.")
```

---

# LAB 1 — The Modern GenAI Stack
**Slot:** Day 1 Morning | After Parts 2-3 | ~45 minutes
**Runtime:** Google Colab — CPU (no GPU needed)
**Theme:** "There are a hundred libraries. Learn the five that matter."

---

## Objectives
By the end of this lab, students will be able to:
1. Navigate the HuggingFace ecosystem (Hub, transformers, tokenizers)
2. Use the `pipeline()` abstraction for zero-setup inference
3. Understand what happens under the hood: tokens → model → logits → text
4. Call a cloud LLM with the OpenAI client using `base_url`
5. Connect LangChain to a cloud backend

## What Students Need
- Google account (for Colab)
- Instructor-provided Groq API key

## Colab Setup Link
`https://colab.research.google.com/` → File → New notebook → rename to `lab1_modern_stack`

## File
`lab1_modern_stack.ipynb`

---

## Step-by-Step Instructions

### Cell 0 — Install (run once at top)

```python
# Cell 0 — Install packages (run this first, takes ~60 seconds)
!pip install -q transformers torch sentence-transformers openai langchain langchain-openai httpx
print("✅ Packages installed")
```

### Cell 1 — Configuration

```python
# Cell 1 — 🔑 CONFIGURATION — paste the shared API key here
# ──────────────────────────────────────────────────────────
GROQ_API_KEY = "gsk_PASTE_KEY_HERE"   # ← instructor provides this
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL  = "llama-3.1-8b-instant"   # fast, good for demos
QUALITY_MODEL  = "llama-3.3-70b-versatile" # better quality, slightly slower
# ──────────────────────────────────────────────────────────
print("Config loaded. Key set:", GROQ_API_KEY[:8] + "..." if GROQ_API_KEY != "gsk_PASTE_KEY_HERE" else "⚠️  KEY NOT SET")
```

### Part A — The HuggingFace Pipeline (15 min)

```python
# Cell 2 — The simplest possible inference
# INSTRUCTOR NOTE: "This is the entire AI industry abstracted into 3 lines."

from transformers import pipeline

# Text generation with GPT-2 — tiny model, runs on CPU in Colab
generator = pipeline("text-generation", model="gpt2")
result = generator("The best way to deploy a language model is", max_new_tokens=40)
print(result[0]["generated_text"])
```

```python
# Cell 3 — Unwrap the pipeline: what's actually happening?
# INSTRUCTOR NOTE: "Let's open the black box."

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

text = "The best way to deploy a language model is"
inputs = tokenizer(text, return_tensors="pt")

print("=== STEP 1: TEXT → TOKENS ===")
print(f"Input text: '{text}'")
print(f"Token IDs:  {inputs['input_ids'].tolist()}")
print(f"Tokens:     {[tokenizer.decode([t]) for t in inputs['input_ids'][0]]}")
print(f"Count:      {inputs['input_ids'].shape[1]} tokens")
```

```python
# Cell 4 — Tokens → Logits → Next token probabilities
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
next_token_logits = logits[0, -1, :]     # last position = prediction for next token
probs = torch.softmax(next_token_logits, dim=-1)

top5 = torch.topk(probs, 5)
print("=== STEP 2: MODEL PREDICTS NEXT TOKEN ===")
print(f"Vocab size: {logits.shape[-1]:,}")
print(f"\nTop 5 next-token candidates:")
for score, idx in zip(top5.values, top5.indices):
    print(f"  '{tokenizer.decode([idx.item()])}' — {score.item()*100:.1f}%")
```

```python
# Cell 5 — Model memory footprint
# INSTRUCTOR NOTE: "This slide connects directly to the 'Memory Formula' you just saw."

def model_size_info(model, label):
    params = sum(p.numel() for p in model.parameters())
    fp32_mb = params * 4 / 1e6
    fp16_mb = params * 2 / 1e6
    int4_mb = params * 0.5 / 1e6
    print(f"\n{label}")
    print(f"  Parameters: {params:,} ({params/1e6:.0f}M)")
    print(f"  Memory FP32: {fp32_mb:.0f} MB  |  FP16: {fp16_mb:.0f} MB  |  INT4: {int4_mb:.0f} MB")
    print(f"  → 7B model FP16 ≈ 14 GB  |  INT4 ≈ 3.5 GB  (that's why quantization matters)")

model_size_info(model, "GPT-2 (124M params)")
```

### Part B — Cloud LLM via Groq (15 min)

```python
# Cell 6 — The KEY pattern: OpenAI client, any backend
# INSTRUCTOR NOTE: "This one pattern unlocks the entire course. One client — swap base_url — any provider."

from openai import OpenAI

client = OpenAI(
    base_url=GROQ_BASE_URL,
    api_key=GROQ_API_KEY
)

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful LLM deployment expert. Be concise."},
        {"role": "user",   "content": "What are the top 3 reasons LLM demos fail in production?"}
    ]
)
print(response.choices[0].message.content)
print(f"\nTokens used: {response.usage.total_tokens}")
```

```python
# Cell 7 — Streaming: watch tokens arrive live
print("Streaming response (tokens appear as generated):\n" + "─"*50)
stream = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[{"role": "user", "content": "Explain the difference between FP16 and INT4 in 4 bullet points."}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n" + "─"*50)
```

```python
# Cell 8 — Switch to a bigger model for quality
# INSTRUCTOR NOTE: "Same code, different model name — this is the entire point of the OpenAI interface."
response_quality = client.chat.completions.create(
    model=QUALITY_MODEL,    # 70B instead of 8B
    messages=[{"role": "user", "content": "In 3 sentences, what is the most important thing to know about LLM serving?"}]
)
print(f"8B model:  {response.choices[0].message.content[:150]}...\n")
print(f"70B model: {response_quality.choices[0].message.content[:150]}...")
```

### Part C — LangChain Integration (15 min)

```python
# Cell 9 — LangChain + Groq backend
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# INSTRUCTOR NOTE: "LangChain also uses the OpenAI interface. One pattern, infinite backends."
llm = ChatOpenAI(
    openai_api_key=GROQ_API_KEY,
    openai_api_base=GROQ_BASE_URL,
    model_name=DEFAULT_MODEL
)

messages = [
    SystemMessage(content="You are an LLM deployment expert. Be concise."),
    HumanMessage(content="What is the difference between vLLM and a simple FastAPI server for LLM inference?")
]
print(llm.invoke(messages).content)
```

```python
# Cell 10 — LangChain prompt templates + chaining
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = ChatPromptTemplate.from_messages([
    ("system", "You are a technical instructor. Explain concepts clearly with one analogy."),
    ("user", "Explain {concept} to someone who {background}.")
])

chain = template | llm | StrOutputParser()

result = chain.invoke({
    "concept": "quantization",
    "background": "has never worked with neural networks before"
})
print(result)
```

## Expected Output
- GPT-2 generates a continuation (nonsensical is fine — it's GPT-2)
- Token IDs and decoded tokens printed clearly
- Top-5 next token probabilities visible
- Groq returns a coherent 3-bullet answer
- Streaming shows tokens arriving one by one
- LangChain chain produces an analogy

## Stretch Goals
1. **Model switch:** Change `DEFAULT_MODEL` to `"mixtral-8x7b-32768"` — observe quality difference
2. **Token counting:** Write a function that takes a string and returns how many tokens it uses via the Groq response's `usage.total_tokens`
3. **Temperature exploration:** Call the same prompt at `temperature=0`, `0.5`, `1.0` — document what changes in the output
4. **Browse the Hub:** Go to `huggingface.co/models?pipeline_tag=text-generation` — find a model fine-tuned for SQL generation

## Instructor Notes
- If students see `"gsk_PASTE_KEY_HERE"` printed: they forgot to paste the key in Cell 1
- Groq free tier limits: 14,400 TPM for 8B, 6,000 TPM for 70B — if rate limited, fall back to 8B
- The streaming demo (Cell 7) is often the most memorable moment — slow down and let everyone watch it

---

# LAB 2 — Loading, Inspecting, and Talking to Models
**Slot:** Day 1 Morning | After Lab 1 | ~45 minutes
**Runtime:** Google Colab — CPU
**Theme:** "Know your model before you ship it."

---

## Objectives
By the end of this lab, students will be able to:
1. Understand model configuration (architecture, layers, attention heads)
2. Compare generation strategies: greedy vs. sampling vs. temperature
3. Use chat templates correctly (why raw prompting breaks instruction models)
4. Build a multi-turn conversation with context window tracking
5. Understand why context overflow is a production bug

## What Students Need
- Google account (for Colab)
- Instructor-provided Groq API key

## File
`lab2_load_inspect_chat.ipynb`

---

## Step-by-Step Instructions

### Cell 0 — Install

```python
!pip install -q transformers torch openai langchain langchain-openai
print("✅ Ready")
```

### Cell 1 — Configuration

```python
# 🔑 CONFIGURATION
GROQ_API_KEY = "gsk_PASTE_KEY_HERE"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL  = "llama-3.1-8b-instant"
print("Config loaded:", GROQ_API_KEY[:8] + "..." if GROQ_API_KEY != "gsk_PASTE_KEY_HERE" else "⚠️  KEY NOT SET")
```

### Part A — Architecture Inspection (15 min)

```python
# Cell 2 — Load a tiny model and inspect its architecture
# INSTRUCTOR NOTE: "We use a small HF model to inspect the architecture.
# This is not for inference — it's for understanding what's inside."
# Qwen2.5-0.5B downloads ~1GB in Colab — takes 2-3 min first time.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"   # Tiny model — safe for Colab CPU
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

config = model.config
print("=== KEY ARCHITECTURE NUMBERS ===")
print(f"Model:           {model_id}")
print(f"Hidden size:     {config.hidden_size} (d_model)")
print(f"Layers:          {config.num_hidden_layers}")
print(f"Attention heads: {config.num_attention_heads}")
print(f"KV heads (GQA):  {getattr(config, 'num_key_value_heads', 'N/A')}")
print(f"Vocab size:      {config.vocab_size:,}")
print(f"Context window:  {config.max_position_embeddings:,} tokens")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {total_params:,} ({total_params/1e9:.2f}B)")
print(f"Memory (BF16): ~{total_params*2/1e9:.1f} GB  |  INT4: ~{total_params*0.5/1e9:.2f} GB")
```

```python
# Cell 3 — Layer structure
print("=== LAYER STRUCTURE (first 20 modules) ===")
for name, module in list(model.named_modules())[:20]:
    print(f"  {name:<40} {type(module).__name__}")
print("  ...")
```

### Part B — Generation Parameters (15 min)

```python
# Cell 4 — Chat with the small local model (no API key needed!)
# INSTRUCTOR NOTE: "We show generation here using the local Colab model.
# This demonstrates what happens under the hood before we move to cloud APIs."

def generate(prompt_text, max_new_tokens=100, **kwargs):
    msgs = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

question = "What are the top 3 challenges in deploying LLMs to production?"
print("GREEDY (deterministic):")
print(f"Run 1: {generate(question, max_new_tokens=60, do_sample=False)[:120]}")
print(f"Run 2: {generate(question, max_new_tokens=60, do_sample=False)[:120]}")
print("(same output both times)")
```

```python
# Cell 5 — Temperature: the single most important knob
# INSTRUCTOR NOTE: "Walk through each temperature slowly. Temperature=0 is greedy."

test_prompt = "Once upon a time, an LLM was deployed to production and"
print("TEMPERATURE COMPARISON (same prompt, 4 different temperatures):\n")
for temp in [0.1, 0.5, 1.0, 1.5]:
    out = generate(test_prompt, max_new_tokens=40, do_sample=True, temperature=temp)
    print(f"  temp={temp}: {out[:100]}...\n")
```

```python
# Cell 6 — Why chat templates matter
# INSTRUCTOR NOTE: "Raw prompting doesn't work on instruction-tuned models.
# The chat template inserts the special tokens the model was trained with."

raw_prompt = "What is quantization?"
formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": raw_prompt}],
    tokenize=False,
    add_generation_prompt=True
)

print("Raw prompt sent to model:")
print(raw_prompt)
print("\nFormatted prompt (with chat template applied):")
print(repr(formatted_prompt))
print("\nThe special tokens (im_start, im_end, etc.) are why the model follows instructions.")
```

### Part C — Multi-Turn Conversation via Cloud API (15 min)

```python
# Cell 7 — Multi-turn chatbot using Groq
# INSTRUCTOR NOTE: "We switch to Groq for the multi-turn demo — much faster at streaming."

from openai import OpenAI

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

class ChatSession:
    """Tracks conversation history — the #1 stateful concept in production chatbots."""
    def __init__(self, system_prompt="You are a helpful assistant."):
        self.history = [{"role": "system", "content": system_prompt}]
    
    def chat(self, user_message, model=DEFAULT_MODEL):
        self.history.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model=model,
            messages=self.history
        )
        answer = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        return answer
    
    def context_length(self):
        """Rough token estimate: ~4 chars per token"""
        total_chars = sum(len(m["content"]) for m in self.history)
        return total_chars // 4
    
    def show_history(self):
        icons = {"system": "🔧", "user": "👤", "assistant": "🤖"}
        for msg in self.history:
            print(f"{icons[msg['role']]} {msg['role'].upper()}: {msg['content'][:100]}{'...' if len(msg['content'])>100 else ''}")
```

```python
# Cell 8 — Run a multi-turn conversation
bot = ChatSession(system_prompt="You are an LLM deployment expert. Be concise — max 2 sentences per answer.")

print(bot.chat("What is quantization?"))
print(f"\n[Context so far: ~{bot.context_length()} tokens]\n")
print("---")
print(bot.chat("And how does it compare to pruning?"))
print(f"\n[Context so far: ~{bot.context_length()} tokens]\n")
print("---")
print(bot.chat("Which should I use first when deploying a 7B model?"))
print(f"\n[Context so far: ~{bot.context_length()} tokens]")

print("\n\nFull conversation history:")
bot.show_history()
```

```python
# Cell 9 — Context window as a production concern
# INSTRUCTOR NOTE: "This is one of the most common production bugs — context overflow."

context_limit = 8192  # llama-3.1-8b context window
current_tokens = bot.context_length()

print("=== CONTEXT WINDOW STATUS ===")
print(f"Used:      ~{current_tokens} tokens")
print(f"Limit:     {context_limit} tokens")
print(f"Remaining: ~{context_limit - current_tokens} tokens")
print(f"Used:      {current_tokens/context_limit*100:.0f}%")
print("\nWhen context fills up in production:")
print("  Option A: Truncate oldest turns (sliding window)")
print("  Option B: Summarize old turns into a compressed system prompt")
print("  Option C: Start a new session with a carry-forward summary")
```

## Expected Output
- Architecture numbers print correctly (hidden_size, num_layers, etc.)
- Greedy decoding produces identical output on both runs
- Temperature 0.1 is coherent and dry; 1.5 is creative/chaotic
- Chat template shows the raw special-token format
- Multi-turn bot accumulates context across 3 turns
- Context window status shows % used

## Stretch Goals
1. **Logprob inspection:** Add `logprobs=True` to the Groq API call and print the top-5 token probabilities for the first word of the response
2. **Model config explorer:** Load the config for `microsoft/phi-4` and `google/gemma-2-2b-it` (no model download needed — just the config). Compare their context windows and hidden sizes
3. **Context overflow simulation:** Loop and keep chatting with the bot until you approach the context limit. What happens?
4. **Sliding window:** Implement `bot.trim_history(max_turns=3)` that keeps only the last 3 user/assistant pairs plus the system prompt

---

# LAB 3 — Quantization and LoRA Fine-Tuning
**Slot:** Day 1 Afternoon | After Parts 4-5 | ~75 minutes
**Runtime:** Google Colab — **T4 GPU required**
**Theme:** "Shrink it. Tune it. Keep the quality."

---

## Objectives
By the end of this lab, students will be able to:
1. Switch Colab to a T4 GPU runtime and verify GPU availability
2. Apply NF4 quantization using bitsandbytes — measure memory and speed
3. Attach LoRA adapters to a quantized model (QLoRA pattern)
4. Run a minimal supervised fine-tuning loop
5. Save LoRA adapters and explain why they're tiny vs. the base model

## What Students Need
- Google account (Colab with T4 GPU)
- No API key needed for this lab (we run locally on Colab GPU)

## ⚠️ RUNTIME SWITCH — Do This First

```
In Colab: Runtime → Change runtime type → T4 GPU → Save
Verify with: !nvidia-smi
```

## File
`lab3_quantize_lora.ipynb`

---

## Step-by-Step Instructions

### Cell 0 — Install + GPU check

```python
# Cell 0 — Install (run after switching to T4 runtime)
!pip install -q transformers torch accelerate bitsandbytes peft trl datasets
print("Packages installed.")

# Verify GPU
import subprocess
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ GPU: {result.stdout.strip()}")
else:
    print("⚠️  No GPU found. Did you switch to T4 runtime? (Runtime → Change runtime type → T4 GPU)")
```

### Part A — Quantization Benchmark (25 min)

```python
# Cell 1 — Load baseline model in FP16
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B loads in ~30 sec on T4

print("Loading FP16 model (baseline)...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
load_time_fp16 = time.time() - start

# Memory in use
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved() / 1e9
print(f"FP16 load time: {load_time_fp16:.1f}s")
print(f"GPU memory allocated: {allocated:.2f} GB")
print(f"GPU memory reserved:  {reserved:.2f} GB")
```

```python
# Cell 2 — Benchmark FP16 inference
def benchmark(model, tokenizer, prompt, n_runs=3, max_new_tokens=80):
    msgs = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        times.append(time.time() - start)
    
    tokens_gen = out.shape[1] - input_len
    avg = sum(times) / len(times)
    text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return {"speed": tokens_gen / avg, "time": avg, "sample": text[:150]}

prompt = "Explain quantization vs pruning in 3 bullet points."
fp16_stats = benchmark(model_fp16, tokenizer, prompt)
fp16_vram = torch.cuda.memory_allocated() / 1e9

print(f"FP16: {fp16_stats['speed']:.1f} tokens/sec, {fp16_vram:.2f} GB VRAM")

del model_fp16
torch.cuda.empty_cache()
```

```python
# Cell 3 — Load NF4 quantized model
# INSTRUCTOR NOTE: "These 4 lines are the canonical NF4 config. Memorize them."
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 4-bit quantization
    bnb_4bit_quant_type="nf4",             # NormalFloat4 — best quality at 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16 even though stored in 4-bit
    bnb_4bit_use_double_quant=True         # Quantize the quantization constants too
)

print("Loading NF4 model...")
start = time.time()
model_nf4 = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)
load_time_nf4 = time.time() - start
nf4_vram = torch.cuda.memory_allocated() / 1e9
print(f"NF4 load time: {load_time_nf4:.1f}s, VRAM: {nf4_vram:.2f} GB")
```

```python
# Cell 4 — Benchmark and compare
nf4_stats = benchmark(model_nf4, tokenizer, prompt)

print("=== QUANTIZATION COMPARISON ===")
print(f"{'Metric':<25} {'FP16':>12} {'NF4':>12}")
print("-"*50)
print(f"{'VRAM (GB)':<25} {fp16_vram:>12.2f} {nf4_vram:>12.2f}")
print(f"{'Tokens/sec':<25} {fp16_stats['speed']:>12.1f} {nf4_stats['speed']:>12.1f}")
print(f"{'Load time (s)':<25} {load_time_fp16:>12.1f} {load_time_nf4:>12.1f}")
print(f"{'Memory reduction':<25} {'baseline':>12} {f'{fp16_vram/nf4_vram:.1f}x smaller':>12}")
print()
print("Sample outputs (same prompt):")
print(f"  FP16: {fp16_stats['sample'][:100]}...")
print(f"  NF4:  {nf4_stats['sample'][:100]}...")
print("\nQuality should be similar — NF4 preserves model quality exceptionally well.")
```

### Part B — LoRA Adapters (25 min)

```python
# Cell 5 — Prepare model for QLoRA training
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_nf4.config.use_cache = False
model_nf4 = prepare_model_for_kbit_training(model_nf4)

# INSTRUCTOR NOTE: "LoRA adds tiny trainable matrices BA where BA = ΔW.
# The base model weights stay frozen. We only train the LoRA matrices."
lora_config = LoraConfig(
    r=16,                      # Rank — bottleneck dimension. Tune this up for harder tasks.
    lora_alpha=32,             # Scaling: typically 2x rank
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model_peft = get_peft_model(model_nf4, lora_config)
model_peft.print_trainable_parameters()
# Expected output: "trainable params: X || all params: Y || trainable%: ~0.7%"
```

```python
# Cell 6 — Build a domain-specific training dataset
# INSTRUCTOR NOTE: "10 examples for demonstration. Real fine-tuning needs 500-5000+."
from datasets import Dataset

training_data = [
    {"instruction": "What is quantization?",
     "response": "Quantization reduces model weight precision (e.g. FP32→INT4), shrinking memory 4-8x with minimal quality loss. NF4 is optimal for LLMs."},
    {"instruction": "What is LoRA?",
     "response": "LoRA freezes base model weights and adds small trainable rank-decomposition matrices BA. Only ~0.7% of parameters are trained."},
    {"instruction": "What is RAG?",
     "response": "RAG retrieves relevant documents from a knowledge base at inference time and injects them into the prompt, grounding the model's answers in facts."},
    {"instruction": "What is vLLM?",
     "response": "vLLM is a serving engine using PagedAttention for efficient KV cache management, enabling 20-30x throughput over naive serving via continuous batching."},
    {"instruction": "What is the difference between fine-tuning and RAG?",
     "response": "Fine-tuning bakes knowledge into weights (good for style/format). RAG retrieves at runtime (good for factual accuracy and updatable knowledge)."},
    {"instruction": "What is PagedAttention?",
     "response": "PagedAttention stores KV cache in non-contiguous memory pages, eliminating fragmentation and enabling efficient multi-user serving."},
    {"instruction": "What is QLoRA?",
     "response": "QLoRA combines 4-bit NF4 base model with 16-bit LoRA adapters, enabling 7B+ fine-tuning on a single consumer GPU."},
    {"instruction": "What is a context window?",
     "response": "The maximum tokens a model processes in one pass — both input and output. Overflowing it causes truncation or errors."},
    {"instruction": "What is chunking in RAG?",
     "response": "Chunking splits documents into segments before embedding. Smaller chunks improve precision; larger chunks preserve context."},
    {"instruction": "What is an embedding?",
     "response": "A dense vector encoding semantic meaning. Similar texts are close in embedding space, enabling similarity search."}
]

def format_example(ex):
    return {"text": tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["instruction"]},
         {"role": "assistant", "content": ex["response"]}],
        tokenize=False, add_generation_prompt=False
    )}

dataset = Dataset.from_list(training_data).map(format_example)
print(f"Dataset: {len(dataset)} examples")
print(f"\nSample formatted example:\n{dataset[0]['text']}")
```

```python
# Cell 7 — Training loop
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True, fp16=False,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    max_seq_length=512,
)

trainer = SFTTrainer(model=model_peft, train_dataset=dataset, args=training_args)

print("Starting QLoRA fine-tuning (T4 GPU — ~3-5 minutes for 10 examples x 3 epochs)...")
trainer.train()
print("✅ Training complete!")
```

### Part C — Save, Inspect, and Reload Adapters (25 min)

```python
# Cell 8 — Save ONLY the LoRA adapter
import os

adapter_path = "./my_lora_adapter"
model_peft.save_pretrained(adapter_path)

print("=== ADAPTER FILES ===")
total_size = 0
for f in os.listdir(adapter_path):
    size = os.path.getsize(os.path.join(adapter_path, f))
    total_size += size
    print(f"  {f}: {size/1e6:.1f} MB")

print(f"\nTotal adapter size: {total_size/1e6:.1f} MB")
print(f"Original model size: ~{nf4_vram*1000:.0f} MB")
print(f"Adapter is {total_size/1e6:.1f} MB vs {nf4_vram*1000:.0f} MB model")
print("→ You ship the adapter (tiny). Users already have the base model.")
```

```python
# Cell 9 — Test the fine-tuned model
from peft import PeftModel

# Reload base + adapter
base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model_tuned = PeftModel.from_pretrained(base, adapter_path)

def tuned_chat(question):
    msgs = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model_tuned.device)
    with torch.no_grad():
        out = model_tuned.generate(**inputs, max_new_tokens=150, do_sample=True,
                                    temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("Fine-tuned model response to 'What is QLoRA?':")
print(tuned_chat("What is QLoRA?"))
```

## Expected Output
- `nvidia-smi` shows T4 GPU with ~15GB total memory
- FP16 vs NF4 comparison: NF4 uses ~4x less VRAM
- `print_trainable_parameters()` shows ~0.7% trainable
- Training shows decreasing loss over 3 epochs
- Adapter directory: ~10MB total vs hundreds of MB model
- Fine-tuned model answers about QLoRA coherently

## Stretch Goals
1. **Rank comparison:** Train with `r=4` vs `r=32` — compare adapter size and answer quality
2. **Merge and unload:** `model_peft.merge_and_unload()` fuses LoRA weights into base model — benchmark inference speed difference
3. **Before/after comparison:** Test the same 3 questions on the base model and the fine-tuned model side-by-side
4. **Export to GGUF (concept):** Research what `llama.cpp convert.py` does and why GGUF is used in production

## Instructor Notes
- If Colab assigns a T4 but runs out of VRAM: use `Qwen/Qwen2.5-0.5B-Instruct` instead of 1.5B
- Colab disconnects after ~90 min of inactivity — remind students to keep the tab active
- The adapter size reveal (Cell 8) is the key demo moment — build up to it

---

# LAB 4 — Serving Models: From Script to API
**Slot:** Day 2 Morning | After Part 6 | ~45 minutes
**Runtime:** Google Colab — CPU
**Theme:** "A model in a notebook is a toy. A model behind an API is a tool."

---

## Objectives
By the end of this lab, students will be able to:
1. Build a FastAPI inference server in Colab using a tunnel URL
2. Call it with the OpenAI Python client
3. Implement streaming SSE responses
4. Understand what vLLM adds on top of a simple FastAPI wrapper
5. Route between providers using the same client (LiteLLM pattern)

## What Students Need
- Google account
- Instructor-provided Groq API key

## ⚠️ Colab + Server Pattern
Running a web server in Colab requires a tunnel. We use `pyngrok` (free).

## File
`lab4_serving.ipynb`

---

## Step-by-Step Instructions

### Cell 0 — Install

```python
!pip install -q fastapi uvicorn pyngrok openai httpx langchain langchain-openai
print("✅ Ready")
```

### Cell 1 — Configuration

```python
# 🔑 CONFIGURATION
GROQ_API_KEY  = "gsk_PASTE_KEY_HERE"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"
print("Config loaded:", GROQ_API_KEY[:8] + "...")
```

### Part A — Build the FastAPI Server (20 min)

```python
# Cell 2 — Write the server file
# INSTRUCTOR NOTE: "We write this to a file, then launch it as a background process."

server_code = '''
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import json, time, uuid, os

app = FastAPI(title="My LLM API", version="0.1.0")

# Backend: Groq (OpenAI-compatible)
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"

backend = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = 500

@app.get("/health")
def health():
    return {"status": "ok", "backend": "groq", "model": DEFAULT_MODEL}

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": DEFAULT_MODEL, "object": "model"}]}

@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    if not request.stream:
        response = backend.chat.completions.create(
            model=request.model, messages=messages,
            temperature=request.temperature, max_tokens=request.max_tokens
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": response.choices[0].message.content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": response.usage.prompt_tokens,
                      "completion_tokens": response.usage.completion_tokens,
                      "total_tokens": response.usage.total_tokens}
        }
    else:
        def generate():
            stream = backend.chat.completions.create(
                model=request.model, messages=messages,
                temperature=request.temperature, stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    data = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0,
                                     "delta": {"content": chunk.choices[0].delta.content},
                                     "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(data)}\\n\\n"
            yield "data: [DONE]\\n\\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
'''

with open("server.py", "w") as f:
    f.write(server_code)
print("✅ server.py written")
```

```python
# Cell 3 — Launch the server with a public tunnel
import subprocess, time, os

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Start server in background
server_proc = subprocess.Popen(["python", "-m", "uvicorn", "server:app", "--port", "8000"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(3)

# Create public tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"✅ Server running!")
print(f"Public URL: {public_url}")
print(f"API Docs:   {public_url}/docs")
print(f"Health:     {public_url}/health")
SERVER_URL = str(public_url)
```

### Part B — Call the Server (15 min)

```python
# Cell 4 — Health check
import httpx
response = httpx.get(f"{SERVER_URL}/health")
print("Health check:", response.json())
```

```python
# Cell 5 — OpenAI client against OUR server
# INSTRUCTOR NOTE: "This is the payoff — any OpenAI client works with our custom server."
from openai import OpenAI

my_client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="not-needed")

response = my_client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "system", "content": "You are an LLM deployment expert."},
        {"role": "user",   "content": "What are 3 advantages of vLLM over a naive FastAPI server?"}
    ]
)
print(response.choices[0].message.content)
print(f"\nTokens: {response.usage}")
```

```python
# Cell 6 — Streaming from our server
print("Streaming:\n" + "─"*50)
stream = my_client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[{"role": "user", "content": "List 5 things that can go wrong when serving LLMs. Be brief."}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n" + "─"*50)
```

```python
# Cell 7 — Provider routing: same client, different backends
# INSTRUCTOR NOTE: "This is LiteLLM's superpower — swap backends by changing base_url."

providers = {
    "Our FastAPI → Groq 8B": {
        "base_url": f"{SERVER_URL}/v1", "api_key": "not-needed", "model": DEFAULT_MODEL
    },
    "Groq Direct 8B": {
        "base_url": GROQ_BASE_URL, "api_key": GROQ_API_KEY, "model": "llama-3.1-8b-instant"
    },
    "Groq Direct 70B": {
        "base_url": GROQ_BASE_URL, "api_key": GROQ_API_KEY, "model": "llama-3.3-70b-versatile"
    }
}

question = "In one sentence: what is PagedAttention?"
print(f"Question: {question}\n")

for provider_name, config in providers.items():
    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    resp = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": question}]
    )
    print(f"[{provider_name}]")
    print(f"  {resp.choices[0].message.content}\n")
```

```python
# Cell 8 — What vLLM adds (conceptual comparison)
print("""
=== OUR FastAPI SERVER vs vLLM ===

Our server (built today):
  ✅ OpenAI-compatible API
  ✅ Streaming support
  ✅ Easy to understand and modify
  ❌ One request at a time (sequential)
  ❌ No KV cache management
  ❌ No GPU memory optimization

vLLM adds:
  ✅ PagedAttention: KV cache in non-contiguous pages (no fragmentation)
  ✅ Continuous batching: serve N users simultaneously on one GPU
  ✅ Throughput: 20-30x improvement over sequential serving
  ✅ AWQ/GPTQ native quantization
  ✅ Tensor parallelism across multiple GPUs

When to use which:
  Our server: Development, testing, small-scale internal tools
  vLLM: Production with real user traffic, cost-sensitive deployments

Production command (GPU server, not Colab):
  python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --quantization awq \\
    --host 0.0.0.0 --port 8000
  → Drop-in replacement for our server. Same client code.
""")
```

```python
# Cell 9 — Cleanup
server_proc.terminate()
ngrok.disconnect(public_url)
print("Server stopped.")
```

## Expected Output
- Server launches with a valid `https://xxxxx.ngrok-free.app` URL
- `/docs` renders Swagger UI (shareable link — students can open on phone)
- OpenAI client returns proper `ChatCompletion` object
- Streaming shows tokens arriving one by one
- Provider routing shows 3 different answers from same code pattern

## Stretch Goals
1. **Middleware logging:** Add a logging middleware to server.py that prints request timestamp, prompt length, and response time
2. **Error handling:** Add a try/except around the Groq call that returns a proper error response if the key is invalid
3. **Multiple models endpoint:** Modify `/v1/models` to list both `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`
4. **LiteLLM:** Run `pip install litellm` and try `litellm --model groq/llama-3.1-8b-instant` — it creates a ready-made proxy server in one command

---

# LAB 5 — Building and Evaluating a RAG Pipeline
**Slot:** Day 2 Morning | After Lab 4 | ~45 minutes
**Runtime:** Google Colab — CPU
**Theme:** "Stop hallucinating. Start retrieving."

---

## Objectives
By the end of this lab, students will be able to:
1. Build a complete 4-stage RAG pipeline (Load → Chunk → Embed → Retrieve/Generate)
2. Apply and compare chunking strategies
3. Use ChromaDB as an in-memory vector store (no setup, no persistence issues)
4. Implement semantic search with sentence-transformers (no API key for embeddings)
5. Evaluate RAG quality using RAGAS

## What Students Need
- Google account
- Instructor-provided Groq API key (for the generation step only)

## Key Design Decision
**Embeddings use `sentence-transformers` (local, free, no API key).** This is production-realistic — embedding 10K documents with a paid API would cost real money. Local embedding models like `all-MiniLM-L6-v2` (46MB) or `BAAI/bge-small-en-v1.5` are industry-standard.

## File
`lab5_rag_pipeline.ipynb`

---

## Step-by-Step Instructions

### Cell 0 — Install

```python
!pip install -q sentence-transformers chromadb langchain langchain-community langchain-openai openai ragas datasets scikit-learn matplotlib
print("✅ Ready (sentence-transformers may take ~2 min to download)")
```

### Cell 1 — Configuration

```python
# 🔑 CONFIGURATION
GROQ_API_KEY  = "gsk_PASTE_KEY_HERE"   # Used only for generation (LLM calls)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"
# No API key needed for embeddings — runs locally via sentence-transformers
print("Config loaded:", GROQ_API_KEY[:8] + "...")
```

### Part A — Load and Chunk (10 min)

```python
# Cell 2 — Create knowledge base (inline — no file system setup needed)
# INSTRUCTOR NOTE: "In production this is PDFs, Confluence, Slack exports.
# We use inline strings so students can swap in their own content later."

knowledge_base = {
    "quantization": """
Quantization reduces model weight precision. Common types: INT8 (1 byte/param, 4x vs FP32),
INT4 (0.5 bytes/param, 8x reduction). NF4 (NormalFloat4) by Tim Dettmers stores values
on a normal distribution — optimal for LLM weights. Combined with double quantization
(quantizing quantization constants), NF4 achieves excellent quality/compression trade-offs.

Post-Training Quantization (PTQ) applies after training — no gradient updates needed.
AWQ analyzes activation distributions to protect important weights from precision loss.
GGUF is the CPU-optimized format used by Ollama and llama.cpp — typically INT4 or INT8.
FlashAttention is complementary: it reduces memory bandwidth during attention computation
and is lossless (no accuracy reduction), unlike quantization.
    """,
    "rag": """
RAG (Retrieval-Augmented Generation) retrieves relevant documents at inference time
and injects them into the model's context, grounding responses in external knowledge.

Four stages: (1) Load source documents. (2) Chunk into segments (256-512 tokens typical).
(3) Embed chunks as dense vectors. (4) At query time: embed query, retrieve similar chunks,
inject into prompt, generate response.

Chunking strategies: fixed-size (simple), sentence-based (respects semantic boundaries),
recursive character (respects document structure), semantic (groups similar sentences).
Hybrid search combines semantic similarity (dense vectors) with BM25 keyword matching.
RAGAS is the standard evaluation framework: faithfulness, answer relevancy, context precision.
    """,
    "lora": """
LoRA (Low-Rank Adaptation) adds small trainable matrices to frozen model weights.
W_new = W + BA where B and A are low-rank matrices (rank r). This trains only ~0.7% of params.

Rank r: 4 for simple style adaptation, 8-16 for moderate task tuning, 32-64 for near full fine-tune.
Alpha is the scaling factor, typically 2x rank. target_modules: usually all attention projections.

QLoRA combines 4-bit NF4 base model with 16-bit LoRA adapters, enabling 7B fine-tuning
on a single consumer GPU (8-12GB VRAM). Adapters are saved separately (~10-100MB)
and loaded on top of the base model at inference time.
    """,
    "serving": """
vLLM uses PagedAttention — KV cache in non-contiguous memory pages (like OS virtual memory).
This eliminates fragmentation and enables efficient serving of many concurrent users.
Continuous batching: new requests are added as compute frees up, not in fixed batches.
GPU utilization improves from ~30% (naive) to 80-90%. 20-30x throughput vs naive serving.

SGLang uses RadixAttention, sharing KV cache prefixes across requests — excellent for RAG
and multi-turn conversations where the system prompt repeats across requests.

Deployment stack: Ollama (dev, GGUF), FastAPI + Groq (prototype), vLLM (GPU production),
TensorRT-LLM (NVIDIA max optimization). All expose OpenAI-compatible endpoints.
    """
}

print(f"Knowledge base: {len(knowledge_base)} topics")
for topic, content in knowledge_base.items():
    print(f"  {topic}: {len(content)} chars")
```

```python
# Cell 3 — Chunking comparison
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def make_docs(kb):
    return [Document(page_content=v.strip(), metadata={"source": k}) for k, v in kb.items()]

raw_docs = make_docs(knowledge_base)

# Two strategies to compare
splitter_small = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
splitter_large = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)

chunks_small = splitter_small.split_documents(raw_docs)
chunks_large = splitter_large.split_documents(raw_docs)

print("=== CHUNKING COMPARISON ===")
print(f"{'Strategy':<25} {'Chunks':>8} {'Avg chars':>12}")
print("-"*47)
avg_small = sum(len(c.page_content) for c in chunks_small) / len(chunks_small)
avg_large = sum(len(c.page_content) for c in chunks_large) / len(chunks_large)
print(f"{'Small (200 chars)':<25} {len(chunks_small):>8} {avg_small:>12.0f}")
print(f"{'Large (400 chars)':<25} {len(chunks_large):>8} {avg_large:>12.0f}")

print(f"\nSample chunk from 'quantization' topic:")
for c in chunks_small:
    if c.metadata["source"] == "quantization":
        print(f"  '{c.page_content[:120]}...'")
        break
```

### Part B — Embed and Store (15 min)

```python
# Cell 4 — Embed with sentence-transformers (FREE, no API key)
# INSTRUCTOR NOTE: "all-MiniLM-L6-v2 is 46MB, runs in seconds on CPU.
# This is production-grade for most use cases — don't pay for embeddings unless you have to."
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model (downloads ~46MB first time)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Test
sample_embedding = embed_model.encode("What is quantization?")
print(f"✅ Embedding dimension: {len(sample_embedding)}")
print(f"   Sample values: {sample_embedding[:5].round(4)}")
```

```python
# Cell 5 — Build ChromaDB vector store (in-memory, no files needed)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# In-memory ChromaDB — works anywhere, no persistence needed for lab
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()  # In-memory — use chromadb.PersistentClient(path=".") for persistence
collection = client.create_collection("llm_course", embedding_function=ef)

# Add all chunks
for i, chunk in enumerate(chunks_large):  # Using larger chunks for better context
    collection.add(
        documents=[chunk.page_content],
        metadatas=[{"source": chunk.metadata["source"]}],
        ids=[f"chunk_{i}"]
    )

print(f"✅ Vector store built: {collection.count()} chunks indexed")
```

```python
# Cell 6 — Semantic search
def search(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0], results["metadatas"][0], results["distances"][0]

query = "How much memory does NF4 quantization save vs FP16?"
docs, metas, distances = search(query)

print(f"Query: '{query}'\n")
print(f"Retrieved {len(docs)} chunks:\n")
for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
    print(f"  [{i+1}] Source: {meta['source']}  |  Distance: {dist:.4f}")
    print(f"       {doc[:150]}...\n")
```

```python
# Cell 7 — Visualize embedding space with PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get all stored embeddings
all_data = collection.get(include=["embeddings", "documents", "metadatas"])
embeddings = np.array(all_data["embeddings"])
sources = [m["source"] for m in all_data["metadatas"]]

# 2D projection
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

# Plot
unique_sources = list(set(sources))
colors = plt.cm.Set1(np.linspace(0, 1, len(unique_sources)))
color_map = dict(zip(unique_sources, colors))

plt.figure(figsize=(9, 6))
for i, (x, y) in enumerate(coords):
    plt.scatter(x, y, color=color_map[sources[i]], s=80, alpha=0.7)
for src, color in color_map.items():
    plt.scatter([], [], color=color, label=src, s=80)

plt.legend(title="Source", bbox_to_anchor=(1, 1))
plt.title("Embedding Space: Knowledge Base Chunks (PCA 2D)\nRelated chunks cluster together")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("embedding_space.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: embedding_space.png  — share this screenshot!")
```

### Part C — Full RAG Chain (15 min)

```python
# Cell 8 — Build RAG chain with LangChain + Groq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    openai_api_key=GROQ_API_KEY,
    openai_api_base=GROQ_BASE_URL,
    model_name=DEFAULT_MODEL,
    temperature=0.1
)

rag_prompt = ChatPromptTemplate.from_template("""You are an expert assistant for an LLM deployment course.
Answer ONLY based on the provided context. If the context lacks the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:""")

def rag_answer(question, n_chunks=3):
    # Retrieve
    docs, metas, _ = search(question, n_results=n_chunks)
    context = "\n\n---\n\n".join([f"[{m['source']}]: {d}" for d, m in zip(docs, metas)])
    
    # Generate
    prompt = rag_prompt.format(context=context, question=question)
    answer = llm.invoke(prompt).content
    
    return answer, metas  # Return answer + sources

question = "What is the memory difference between NF4 and FP32 quantization?"
answer, sources = rag_answer(question)
print(f"Q: {question}")
print(f"A: {answer}")
print(f"\nSources: {[s['source'] for s in sources]}")
```

```python
# Cell 9 — RAG vs No-RAG comparison
test_q = "What is double quantization and what does it save?"

print("WITHOUT RAG (model's training knowledge alone):")
direct = llm.invoke(test_q).content
print(direct[:300])

print("\n" + "─"*60)
print("WITH RAG (grounded in our knowledge base):")
rag, _ = rag_answer(test_q)
print(rag)
print("\n(RAG answer cites specific details from the knowledge base)")
```

```python
# Cell 10 — Multi-question evaluation
test_questions = [
    "What chunking size should I use for RAG?",
    "How does LoRA reduce training costs?",
    "What is the difference between vLLM and SGLang?",
    "Why is NF4 better than INT4 for LLMs?"
]

print("=== RAG PIPELINE — BATCH EVALUATION ===\n")
for q in test_questions:
    ans, srcs = rag_answer(q)
    print(f"Q: {q}")
    print(f"A: {ans[:150]}...")
    print(f"Sources: {[s['source'] for s in srcs]}\n")
```

```python
# Cell 11 — RAGAS evaluation (optional — needs model judge)
# INSTRUCTOR NOTE: "RAGAS uses an LLM to evaluate faithfulness and relevance.
# This is the LLM-as-Judge pattern. We use Groq's 70B as the judge."
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

judge_llm = ChatOpenAI(
    openai_api_key=GROQ_API_KEY,
    openai_api_base=GROQ_BASE_URL,
    model_name="llama-3.3-70b-versatile"   # Use the bigger model as judge
)

eval_questions = [
    "What is NF4 quantization?",
    "What is QLoRA?",
    "What is PagedAttention?"
]

eval_data = []
for q in eval_questions:
    docs, metas, _ = search(q)
    answer, _ = rag_answer(q)
    eval_data.append({
        "question": q,
        "answer": answer,
        "contexts": docs,
        "ground_truth": answer  # Proxy — in production use human-labeled references
    })

dataset = Dataset.from_list(eval_data)

try:
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print("=== RAGAS SCORES ===")
    print(f"Faithfulness:     {results['faithfulness']:.3f}  (is the answer grounded in context?)")
    print(f"Answer Relevancy: {results['answer_relevancy']:.3f}  (does the answer address the question?)")
    print("\nTarget: > 0.8 is good, > 0.9 is excellent")
except Exception as e:
    print(f"RAGAS note: {e}")
    print("Manual evaluation:")
    for item in eval_data:
        print(f"  Q: {item['question']}")
        print(f"  A: {item['answer'][:120]}...\n")
```

## Expected Output
- Chunking comparison shows more/fewer chunks per strategy
- Embedding dimension: 384 (all-MiniLM-L6-v2)
- PCA plot shows distinct clusters per topic
- RAG answers cite specific numbers from the knowledge base
- RAG vs No-RAG comparison shows grounded vs generic answers
- RAGAS scores (if available) or manual printout

## Stretch Goals
1. **BM25 hybrid search:** Install `rank-bm25`, create a `BM25Okapi` index over chunk texts, combine with semantic scores using a weighted sum
2. **Larger knowledge base:** Add 5 URLs of LLM papers/articles using `langchain_community.document_loaders.WebBaseLoader` — re-embed and test retrieval quality
3. **Better embedding model:** Swap `all-MiniLM-L6-v2` for `BAAI/bge-small-en-v1.5` — both are free/local. Compare retrieval quality on the test questions.
4. **HyDE:** Generate a hypothetical answer to the question first, then embed THAT instead of the raw question — compare retrieved chunks

---

# LAB 6 — Deploy and Showcase: From API to User Interface
**Slot:** Day 2 Afternoon | After Part 8 | ~45 minutes
**Runtime:** Google Colab — CPU
**Theme:** "If it doesn't have a UI, it's not a product."

---

## Objectives
By the end of this lab, students will be able to:
1. Build a Gradio RAG chat interface that works from Colab
2. Add streaming token output to the UI
3. Display retrieved sources alongside each answer (transparency)
4. Generate a public share link (show the app on a phone)
5. Add basic observability: per-query latency logging

## What Students Need
- Google account
- Instructor-provided Groq API key

## ⚠️ Important
This lab requires Lab 5's in-memory ChromaDB to be active in the SAME Colab session. Run Lab 5 Cells 0-8 first, or re-run them at the top of this notebook.

## File
`lab6_gradio_rag_app.ipynb`

---

## Step-by-Step Instructions

### Cell 0 — Install

```python
!pip install -q gradio sentence-transformers chromadb langchain langchain-openai openai
print("✅ Ready")
```

### Cell 1 — Configuration + Re-build knowledge base

```python
# 🔑 CONFIGURATION
GROQ_API_KEY  = "gsk_PASTE_KEY_HERE"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"
print("Config:", GROQ_API_KEY[:8] + "...")

# Rebuild the knowledge base from Lab 5 (copy these cells if not already run)
# ──────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

knowledge_base = {
    "quantization": "Quantization reduces model weight precision. NF4 (NormalFloat4) achieves 8x compression vs FP32 with excellent quality. Double quantization saves another 0.4 bits/param. AWQ protects important weights from precision loss. GGUF is the CPU format used by Ollama.",
    "rag": "RAG retrieves relevant documents at inference time and injects them into the prompt. Four stages: Load, Chunk, Embed, Retrieve+Generate. Hybrid search combines semantic and BM25. RAGAS evaluates faithfulness and answer relevancy.",
    "lora": "LoRA adds trainable rank-r matrices BA to frozen weights. QLoRA combines NF4 base with 16-bit LoRA adapters. Rank 16 is a good starting point. Adapters are 10-100MB, saved separately from the base model.",
    "serving": "vLLM uses PagedAttention for non-contiguous KV cache pages — 20-30x throughput vs naive serving. Continuous batching serves N concurrent users. SGLang RadixAttention shares KV prefixes across requests. All expose OpenAI-compatible endpoints."
}

ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma = chromadb.Client()
collection = chroma.create_collection("lab6_kb", embedding_function=ef)

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
chunks = splitter.split_documents([Document(page_content=v, metadata={"source": k}) for k, v in knowledge_base.items()])

for i, chunk in enumerate(chunks):
    collection.add(documents=[chunk.page_content], metadatas=[{"source": chunk.metadata["source"]}], ids=[f"c{i}"])

print(f"✅ Knowledge base: {collection.count()} chunks indexed")
```

### Part A — Core RAG Function with Streaming (15 min)

```python
# Cell 2 — RAG retrieval + streaming generation
from openai import OpenAI
import time, json
from datetime import datetime

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

RAG_PROMPT = """You are an expert assistant for an LLM deployment course.
Answer ONLY based on the provided context. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

def retrieve(query, n=3):
    results = collection.query(query_texts=[query], n_results=n)
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas

def rag_stream(question):
    """Generator: yields (partial_answer, sources_md, latency_ms)"""
    start = time.time()
    
    # Retrieve
    docs, metas = retrieve(question)
    context = "\n\n".join([f"[{m['source']}]: {d}" for d, m in zip(docs, metas)])
    
    # Build sources markdown
    sources_md = "**📚 Retrieved Sources:**\n"
    for i, (d, m) in enumerate(zip(docs, metas)):
        sources_md += f"\n**{i+1}. {m['source']}**\n_{d[:100]}..._\n"
    
    # Stream generation
    prompt = RAG_PROMPT.format(context=context, question=question)
    full_answer = ""
    
    for chunk in client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        if chunk.choices[0].delta.content:
            full_answer += chunk.choices[0].delta.content
            yield full_answer, sources_md, None
    
    latency_ms = int((time.time() - start) * 1000)
    sources_md += f"\n\n_⏱ {latency_ms}ms_"
    yield full_answer, sources_md, latency_ms

# Quick test
print("Testing RAG stream...")
for answer, sources, latency in rag_stream("What is NF4 quantization?"):
    pass  # Consume the generator
print(f"Answer: {answer[:150]}...")
print(f"Latency: {latency}ms")
```

### Part B — Gradio UI (20 min)

```python
# Cell 3 — Full Gradio RAG chatbot
import gradio as gr

# ── Query log (in-memory for this session) ──
query_log = []

def respond(message, chat_history):
    """Called by Gradio on each message. Streams tokens to the UI."""
    if not message.strip():
        yield "", chat_history, ""
        return
    
    chat_history = chat_history + [(message, "")]
    sources_display = ""
    
    for answer_so_far, sources_so_far, latency in rag_stream(message):
        chat_history[-1] = (message, answer_so_far)
        sources_display = sources_so_far
        yield "", chat_history, sources_display
    
    # Log the query
    query_log.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "query": message[:60],
        "latency_ms": latency,
        "sources": sources_display[:50]
    })

def get_log():
    if not query_log:
        return "No queries yet."
    rows = [f"| {e['time']} | {e['query']} | {e['latency_ms']}ms |" for e in query_log[-10:]]
    return "| Time | Query | Latency |\n|---|---|---|\n" + "\n".join(rows)

# ── Build the UI ──
with gr.Blocks(title="LLM Course Assistant", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🤖 LLM Deployment Course Assistant
    Ask anything about **quantization, RAG, LoRA, or serving**. All local embeddings + Groq generation.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=420, show_copy_button=True)
            
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask about quantization, RAG, LoRA, vLLM...",
                                 show_label=False, scale=5, container=False)
                send_btn = gr.Button("Send ▶", variant="primary", scale=1)
            
            gr.Examples(
                examples=[
                    "What is the memory savings from NF4 vs FP32?",
                    "When should I use RAG vs fine-tuning?",
                    "What is QLoRA and how does it work?",
                    "What makes vLLM faster than a naive FastAPI server?"
                ],
                inputs=msg
            )
        
        with gr.Column(scale=2):
            sources_box = gr.Markdown(
                "*Sources will appear here after your first question.*",
                label="Retrieved Context"
            )
            
            with gr.Accordion("Query Log", open=False):
                log_display = gr.Markdown("No queries yet.")
                refresh_btn = gr.Button("Refresh Log", size="sm")
    
    # Wire up
    send_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot, sources_box])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot, sources_box])
    refresh_btn.click(get_log, outputs=log_display)

print("Launching Gradio app...")
```

```python
# Cell 4 — Launch with a public share link
# INSTRUCTOR NOTE: "share=True generates a public https://xxxxx.gradio.live link.
# Students can open this on their phones — live demo!"

demo.launch(
    share=True,        # Creates public URL — works from Colab
    debug=False,
    quiet=True
)

# The public URL will print in the output below.
# Share it with the class — anyone can open it!
```

```python
# Cell 5 — Test the app via its API (Gradio has REST API built in)
import httpx, time

# Wait for app to start
time.sleep(3)

# Gradio exposes a REST API automatically
test_response = httpx.post(
    "http://127.0.0.1:7860/run/predict",
    json={"data": ["What is PagedAttention?", []]},
    timeout=30.0
)
if test_response.status_code == 200:
    print("✅ App is running and responding to API calls")
else:
    print("App is running — access via the public URL printed above")
```

```python
# Cell 6 — Show query log
import pandas as pd

if query_log:
    df = pd.DataFrame(query_log)
    print("=== QUERY LOG ===")
    print(df.to_string(index=False))
else:
    print("Ask some questions in the UI first, then run this cell.")
```

## Expected Output
- Gradio app launches with a public `gradio.live` URL
- Chat interface streams tokens as they're generated
- Right panel shows retrieved source chunks with source labels
- Example questions populate the input with one click
- Query log tracks time, query text, and latency
- The app is accessible on a phone via the share link

## Stretch Goals
1. **Upload your own documents:** Add a `gr.File` component; on upload, chunk, embed, and add to the collection
2. **Model selector:** Add a `gr.Radio` to switch between `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`
3. **Dark mode:** Swap `gr.themes.Soft()` for `gr.themes.Monochrome()` — try other built-in themes
4. **HuggingFace Spaces:** Go to `huggingface.co/spaces` → Create new Space → Gradio → paste your code. Your app becomes permanently hosted (free)!

## Instructor Notes
- If `share=True` doesn't work (firewall): use `server_name="0.0.0.0"` + pyngrok from Lab 4 instead
- The phone demo is the most memorable moment of the day — announce it before launching
- HuggingFace Spaces as a stretch goal gives students a permanent public URL they can share on LinkedIn

---

# CAPSTONE — End-to-End LLM Deployment System
**Slot:** Day 2 Afternoon | Final 90 minutes
**Runtime:** Google Colab — CPU (or T4 GPU for Track B)
**Theme:** "This is the thing you take home and show your team."

---

## Overview
The Capstone integrates Labs 1-6 into a single deployable system. Students choose from two tracks.

---

## TRACK A — Domain RAG Assistant (Recommended for most students)
*Build: Your own documents → RAG pipeline → Gradio UI → Share URL → Present live*

### What Makes It "Your Own"
Students bring 3-5 documents from their own domain (or choose from provided options):
- Their company's public documentation
- Wikipedia articles on a topic they know well
- Academic paper abstracts in their field
- Product specs, support articles, research notes

### Provided Document Sources (no auth required)
```
Technical:
  - Hugging Face documentation pages (WebBaseLoader)
  - PyPI package README files
  - RFC documents (ietf.org)

Domain Options:
  - PubMed abstracts (medical)
  - SEC EDGAR filings (finance)
  - arXiv paper abstracts (ML/science)
  - Cookbooks/recipes (food tech)
```

### Capstone A Notebook

```python
# capstone_a_domain_rag.ipynb

# Cell 0 — Install
!pip install -q gradio sentence-transformers chromadb langchain langchain-community langchain-openai openai pypdf

# Cell 1 — Configuration
GROQ_API_KEY  = "gsk_PASTE_KEY_HERE"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"

# Cell 2 — Load YOUR documents
# Option A: Paste text inline (fastest)
my_documents = {
    "topic_1": "Your first document content here...",
    "topic_2": "Your second document content here...",
    "topic_3": "Your third document content here...",
}

# Option B: Load from the web (no file upload needed)
from langchain_community.document_loaders import WebBaseLoader
urls = [
    "https://huggingface.co/docs/transformers/index",    # Example — replace with your URLs
    "https://docs.vllm.ai/en/latest/index.html",
]
loader = WebBaseLoader(urls)
web_docs = loader.load()
print(f"Loaded {len(web_docs)} pages from the web")

# ── [Use the same pipeline from Lab 5: chunk → embed → store → retrieve → generate] ──
# ── [Use the same Gradio UI from Lab 6] ──
# ── Copy and adapt cells from Labs 5 and 6 ──

# The capstone should feel like YOUR product, not a course demo.
```

---

## TRACK B — Fine-Tuned Model Showcase (GPU required)

*Build: Custom dataset → QLoRA fine-tuning → Compare to base → Gradio demo*

### What to Build
1. Curate 20-50 instruction examples for a specific task the base model is weak at
2. Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` with QLoRA (small enough for reliable T4 training)
3. Create a side-by-side demo: base model vs. fine-tuned model on 5 test prompts
4. Build a Gradio interface with "Before / After" comparison columns

### Capstone B Notebook Structure

```python
# capstone_b_finetuning.ipynb

# Step 1: Identify the task (examples: SQL generation, JSON formatting,
#         formal letter writing, medical terminology simplification,
#         code documentation, recipe generation from ingredients)

# Step 2: Create training data (20-50 examples minimum)
training_data = [
    {"instruction": "TASK-SPECIFIC PROMPT", "response": "IDEAL RESPONSE"},
    # ... more examples
]

# Step 3: QLoRA fine-tuning (reuse Lab 3 train.py pattern)
# Step 4: Evaluate before/after on 5 held-out test examples
# Step 5: Gradio UI with before/after columns

import gradio as gr

def compare(question):
    base_answer = chat_base_model(question)      # Base model
    tuned_answer = chat_finetuned_model(question) # Fine-tuned model
    return base_answer, tuned_answer

with gr.Blocks() as demo:
    gr.Markdown("# Before / After Fine-Tuning")
    question = gr.Textbox(label="Test prompt")
    with gr.Row():
        base_output  = gr.Textbox(label="Base Model (Qwen 0.5B)")
        tuned_output = gr.Textbox(label="Fine-Tuned Model")
    compare_btn = gr.Button("Compare")
    compare_btn.click(compare, inputs=question, outputs=[base_output, tuned_output])
```

---

## Capstone Presentation (5 min per student)

```
1. PROBLEM (30 sec)
   "I built a [domain] assistant for [use case]."

2. ARCHITECTURE (60 sec)
   Whiteboard or simple diagram:
   [Documents] → [Chunks] → [ChromaDB] → [Retriever] → [Groq LLM] → [Gradio UI]

3. LIVE DEMO (2 min)
   Ask 3 questions live. Show retrieved sources. Show one failure case.
   BONUS: Open the share link on your phone.

4. LESSONS LEARNED (90 sec)
   What chunking strategy did you use and why?
   What broke first?
   What would you do differently?
```

### Evaluation Rubric

| Criterion | 1 | 3 | 5 |
|:---|:---|:---|:---|
| **Working system** | Errors during demo | Works but slow/limited | Smooth, handles edge cases |
| **Own domain** | Generic demo dataset | Student-provided docs | High-value domain, clear use case |
| **RAG quality** | Hallucinating | Mostly grounded | Faithful, cites sources |
| **Code quality** | Single messy notebook | Organized cells | Reusable functions, config cell |
| **Reflection** | "It works" | Discusses one tradeoff | Discusses chunking, evaluation, failure mode |

---

# APPENDIX A — Quick Reference

## Model Quick Reference (Groq Free Tier)

| Model | Speed | Quality | Best For |
|:---|:---|:---|:---|
| `llama-3.1-8b-instant` | Very fast | Good | All demos, default choice |
| `llama-3.3-70b-versatile` | Moderate | Excellent | Quality demos, RAGAS judge |
| `mixtral-8x7b-32768` | Fast | Good | Long context (32K window) |
| `llama-3.1-70b-versatile` | Moderate | Excellent | Complex reasoning |

## Groq Rate Limits (Free Tier)

| Model | Tokens/minute | Requests/minute |
|:---|:---|:---|
| 8B models | 14,400 | 30 |
| 70B models | 6,000 | 30 |

**If rate-limited:** Switch to `llama-3.1-8b-instant`, wait 60 seconds, or use a backup key.

## Embedding Models (Local, Free, No API Key)

| Model | Size | Dimension | Best For |
|:---|:---|:---|:---|
| `all-MiniLM-L6-v2` | 46 MB | 384 | Default — fast, balanced |
| `BAAI/bge-small-en-v1.5` | 134 MB | 384 | Better quality |
| `BAAI/bge-base-en-v1.5` | 438 MB | 768 | Production quality |

## Common Errors in Colab

```
ERROR: "gsk_PASTE_KEY_HERE" — AuthenticationError
FIX: Paste the instructor-provided Groq key in Cell 1 and re-run

ERROR: No GPU found
FIX: Runtime → Change runtime type → T4 GPU → Save → re-run from Cell 0

ERROR: "ResourceExhausted" from Groq (rate limit)
FIX: Wait 60 seconds; switch to llama-3.1-8b-instant; use backup key

ERROR: Colab session disconnected
FIX: Re-run from Cell 0; all in-memory data (ChromaDB) resets — re-run Lab 5 setup cells

ERROR: "module not found"
FIX: Re-run Cell 0 (the !pip install cell) — Colab doesn't persist installs across sessions

ERROR: ngrok tunnel expired (Lab 4)
FIX: Re-run the ngrok cell to get a new tunnel URL

ERROR: Gradio share link not working
FIX: Use server_name="0.0.0.0" + pyngrok instead; or deploy to HuggingFace Spaces
```

---

# APPENDIX B — Google Colab Tips for Students

```
Starting a new lab:
  1. Go to colab.research.google.com
  2. File → New notebook (or open the shared link)
  3. Run Cell 0 (pip install) first — always
  4. Paste the API key in Cell 1 — always

GPU runtime (Lab 3 only):
  Runtime → Change runtime type → T4 GPU → Save
  Verify: !nvidia-smi

Keeping session alive:
  Colab disconnects after ~90 min of no interaction
  Move your mouse or run a cell every hour
  If disconnected: re-run ALL cells from the top (in-memory data is lost)

Saving your work:
  File → Save a copy in Drive (Ctrl+S)
  Your notebooks auto-save but outputs reset on reconnect

Sharing your Gradio app permanently (free):
  1. Go to huggingface.co/spaces
  2. Create new Space → SDK: Gradio
  3. Paste your lab6 code into app.py
  4. Add a requirements.txt
  5. Your app gets a permanent URL: username.hf.space/space-name
```

---

# APPENDIX C — Suggested Document Sources for Capstone Track A

```python
# These all work with WebBaseLoader — no file upload needed

# HuggingFace documentation
urls_ml = [
    "https://huggingface.co/docs/transformers/index",
    "https://huggingface.co/docs/peft/index",
    "https://huggingface.co/docs/trl/index",
]

# vLLM documentation
urls_serving = [
    "https://docs.vllm.ai/en/latest/index.html",
    "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html",
]

# Gradio documentation
urls_ui = [
    "https://www.gradio.app/docs/gradio/blocks",
    "https://www.gradio.app/docs/gradio/chatbot",
]

# arXiv abstracts (ML papers)
urls_papers = [
    "https://arxiv.org/abs/2305.14314",   # QLoRA paper
    "https://arxiv.org/abs/2309.06180",   # Mistral paper
]

# Usage example
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(urls_ml)
docs = loader.load()
print(f"Loaded {len(docs)} pages, {sum(len(d.page_content) for d in docs)} chars total")
```

---

*End of claude_labs_master.md*
*Generated by Claude | May 2026 | Mastering LLM Deployment — 2-Day Intensive*
*Designed for Google Colab free tier | Instructor-provided Groq API key | No local installation required*
