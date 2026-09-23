# Lab 4 — Quantization and QLoRA Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/04_Quantize_LoRA/lab4_quantize_lora.ipynb)

**Day 1 Afternoon | ~90 minutes | T4 GPU required ⚡ | No API key**

> **Enable the GPU before you open the notebook:**
> Runtime → Change runtime type → Hardware accelerator → **T4 GPU** → Save

---

## Coming from Lab 3

Lab 3 was arithmetic on a CPU. You computed what a model costs to hold — `params × 2` for BF16, `params × 0.5` for INT4 — and estimated how many users would fit on a 16 GB GPU alongside their KV caches.

Lab 4 tests those numbers on real hardware, then goes further: you take a model squeezed into 4 bits and **train** it anyway. The `q_proj` tensor you measured in Lab 3 comes back in Part A as the thing you quantize by hand.

---

## Purpose

A 7B model in FP32 needs about 28 GB of VRAM. A free Colab T4 has 15. This lab covers the two techniques that close that gap — **quantization** to make the model fit, and **LoRA** to make fine-tuning affordable — and it explains the mechanism of each rather than just the API call.

---

## What You Will Build

**Part A — What 4 bits actually means (~20 min, CPU)**
Four bits gives you sixteen values. Where do they come from, where are they placed, and how many weights share one scale? You quantize a real Qwen weight tensor by hand and discover that a single 18-sigma outlier destroys a whole-tensor scale, that per-64-weight blocks are roughly ten times more accurate at the same bit width, that "4-bit" is really 4.25 bits, and that NF4's quantile-spaced levels beat evenly spaced ones by about 20% error for free. By the end, every argument of `BitsAndBytesConfig` maps to something you measured.

**Part B — Measure it on the T4 (~20 min)**
Load `Qwen2.5-1.5B-Instruct` in FP16 and in NF4. Compare peak VRAM, tokens/sec, and quality. Memory drops about 3×. **Speed usually drops too** — and the lab explains why, because the weights are dequantized to bf16 on every matmul. Quantization buys fit, not speed, and that distinction decides whether you should use it at all.

**Part C — LoRA and QLoRA (~25 min, CPU then GPU)**
Why full fine-tuning needs ~16 bytes per parameter and therefore won't fit. Why ΔW is low rank. You size the A and B matrices by hand, prove the adapter contributes *exactly zero* at initialization, and see what rank trades off. Then attach real adapters to the quantized model, confirm `lora_B` is all zeros inside an actual layer, and train with `SFTTrainer`. Includes why `use_cache=False` during training does not contradict Lab 3.

**Part D — Ship the artifact (~20 min)**
Save the adapter, write a model card, reload a clean base, and compare base against tuned on the same three questions. Then merge into an FP16 base for export and review the GGUF/Ollama path. Three artifacts, and when each one is right.

**Bonus — Pruning, and why it loses (~8 min)**
Zero the smallest 30% of every linear weight, then save the model and watch the file size not move. A zero in a dense tensor still costs two bytes. Closes with where sparsity actually pays off: Mixture of Experts.

---

## Critical Points

**Quantization is a memory technique, not a speed technique.** NF4 is frequently *slower* than FP16 on a T4 because every matmul dequantizes the weights back to bf16 first. What you buy is fit: a bigger model, more concurrent users, or more room for the KV cache. If the model already fits comfortably in FP16, quantizing may make things worse.

**Block size is the whole game.** Trained weights cluster within one sigma of zero, but outliers sit eighteen sigma out. One shared scale across a tensor lets a single outlier flatten everything else onto two or three levels. A scale per 64 weights contains the damage — measurably about 10× less error at the same bit width.

**"4-bit" is 4.25 bits.** Every block stores a scale alongside it. Double quantization compresses those scales and recovers roughly 0.4 bits per weight.

**NF4's levels are not evenly spaced.** They sit at the quantiles of a normal distribution, tightly packed near zero where the weights are and spread out at the edges where they are not. Same storage, about 20% less error.

**LoRA adapters start as a no-op.** `lora_B` is initialized to zeros, so `x·A·B` is zero and the model begins bit-for-bit identical to the base you already trusted. Every subsequent change is something the optimizer chose.

**QLoRA composes because frozen weights don't need precision.** No gradients flow to the base, so put it in 4 bits. Gradients and Adam state live only in the tiny adapters, so keep those in bf16. Precision goes where the optimization happens.

**The adapter is the deliverable.** Small enough to version in git-lfs, attach to a pull request, roll back, and A/B test. One base model in VRAM can serve many adapters swapped per request. Merge only when the target runtime demands a single directory.

**Fine-tuning changes how a model answers more than what it knows.** Ten examples over three epochs will visibly shift style and length. It will not make the model factually reliable — expect confidently wrong tuned answers, and note that this is what fine-tuning is and is not for. Facts that change are a retrieval problem (Lab 6).

---

## Key Terms

| Term | Definition |
|------|-----------|
| Quantization | Storing weights in fewer bits (16-bit → 4-bit) to reduce memory |
| Block-wise quantization | One scale per small group of weights (typically 64) instead of one per tensor, so outliers only damage their own block |
| Scale | The per-block multiplier that maps quantized levels back to real weight values |
| NF4 | NormalFloat4 — sixteen levels placed at normal-distribution quantiles rather than evenly |
| Double quantization | Quantizing the block scales themselves, recovering ~0.4 bits per weight |
| `bnb_4bit_compute_dtype` | The dtype weights are dequantized *to* for the actual matmul. Why 4-bit is not faster. |
| PTQ | Post-Training Quantization — quantize an already-trained model, no retraining |
| LoRA | Low-Rank Adaptation — freeze W, train two thin matrices A and B whose product approximates ΔW |
| Rank `r` | The bottleneck dimension of A and B. 8–16 for style and format, 32–64 for new behaviour. |
| `lora_alpha` | Scaling factor; the adapter's output is multiplied by `alpha/r`. Conventionally `2r`. |
| QLoRA | 4-bit frozen base + bf16 LoRA adapters trained on top |
| `prepare_model_for_kbit_training` | Casts layer norms to fp32, enables gradient checkpointing, lets gradients reach the adapters |
| SFTTrainer | Supervised fine-tuning loop from `trl` |
| Adapter | The trained LoRA weights, saved separately from the base |
| Merged model | A plain model directory after `merge_and_unload()` folds the adapter into the weights |
| GGUF | The local runtime format for llama.cpp, Ollama and LM Studio |
| Magnitude pruning | Zeroing the smallest weights. Does not shrink a dense file or speed anything up without sparse kernels. |
| MoE | Mixture of Experts — structured sparsity that does pay off, activating only a few expert blocks per token |

---

## Before This Lab

**Switch to a T4 first.** If you load weights on CPU and switch afterwards, you lose the runtime and start over.

Downloads: `Qwen2.5-0.5B-Instruct` (~1 GB, for the Part A quantization demo — the same file as Lab 3) and `Qwen2.5-1.5B-Instruct` (~3 GB, loaded twice in Part B). No API key.

If you hit a VRAM error, change `MODEL_ID` in cell B1 to the 0.5B model and continue; every lesson still lands.

Parts A and C begin on the **CPU** with small tensors, then move to the GPU. If Colab reclaims your GPU mid-lab, the concept cells still run.

---

## Next

[Lab 5 — Serving API](../05_Serving_API/README.md) is Day 2 morning, back on CPU. You will put a model behind an OpenAI-compatible FastAPI server — the `base_url` swap from Lab 1A, except the URL is yours — and expose it with ngrok. Add a Colab Secret `NGROK_AUTH_TOKEN` before class if you can.

The thread: Lab 3 told you what a model costs to hold, Lab 4 made it fit and taught it something, Lab 5 puts it behind an endpoint.

Want to serve the model you just trained? [Bonus 03](../Bonus/03_vllm_serving.ipynb) runs it with vLLM on a T4. Before this runtime ends, run `!zip -r my_lora_adapter.zip my_lora_adapter` and download the zip; Bonus 03 asks for it.
