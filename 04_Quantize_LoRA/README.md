# Lab 3 — Quantization and QLoRA Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/03_Quantize_LoRA/lab3_quantize_lora.ipynb)

**Day 1 Afternoon | ~75 minutes | T4 GPU required ⚡**

> **Enable GPU before opening the notebook:**
> Runtime → Change runtime type → Hardware accelerator → **T4 GPU** → Save

---

## Purpose

A 7-billion-parameter model in full 32-bit precision needs ~28 GB of VRAM. A T4 GPU (Colab free tier) has 15 GB. This lab teaches you the two techniques that bridge that gap: **quantization** (load the existing model in fewer bits) and **QLoRA** (fine-tune it without touching the original weights). Together, they let you run and adapt large models on hardware you actually have.

---

## The Problem This Lab Solves

| Scenario | Challenge |
|----------|-----------|
| You want to run a 7B model for inference | Needs ~14 GB VRAM in FP16 — barely fits a T4 |
| You want to fine-tune that 7B model | Full fine-tuning needs 3–4× the model size in VRAM for gradients + optimizer state |
| You want to deploy multiple fine-tuned variants | A full fine-tuned copy per task = 14 GB × N stored and loaded |

Quantization and LoRA each solve part of this. QLoRA solves all of it.

---

## What You Will Build

**Part A — Quantization Benchmark**
Load `Qwen2.5-1.5B-Instruct` twice: once in FP16 (baseline) and once in NF4 (4-bit). Measure load time, VRAM usage, and tokens/sec. You'll see the concrete numbers behind the "4-bit is ~4× smaller" claim — and measure the actual quality impact.

**Part B — LoRA Mechanics**
Attach LoRA adapters to the quantized model. Print the trainable parameter count — it should be around 0.7% of total parameters. Understand the math behind why this works.

**Part C — QLoRA Fine-Tuning**
Train on a 10-example LLM deployment Q&A dataset using `SFTTrainer`. Save the adapter (expect ~10 MB vs ~3 GB for the full model). Reload it and compare the fine-tuned model's answers to the base model.

**Bonus — Magnitude Pruning**
Apply global unstructured pruning at 30% sparsity. Measure before/after sparsity and observe the quality trade-off.

---

## Critical Points

**Quantization is lossy — but often not meaningfully so.** NF4 (4-bit NormalFloat) is designed to minimize the information loss when reducing from 16-bit. For most tasks, the accuracy drop is small enough that the memory savings are clearly worth it.

**The precision hierarchy:**
```
FP32  →  FP16  →  INT8  →  INT4 (NF4)
32 bit   16 bit   8 bit    4 bit
~4×      ~2×      baseline  ~2× smaller than INT8
```

**LoRA does not change the base model weights.** It freezes the original model and trains two small matrices (A and B) whose product approximates the weight update: `W_new = W + B×A`. Rank `r` controls the size of these matrices — higher rank = more capacity = more trainable parameters.

**QLoRA = NF4 quantized base + FP16 LoRA adapters.** The base model is in 4-bit (for memory), but the adapter training happens in 16-bit (for precision). This is the key insight: you get the memory savings of quantization without sacrificing the training quality.

**Adapter files are tiny and swappable.** A fine-tuned LoRA adapter for a 1.5B model is ~10 MB. You can store dozens of task-specific adapters and load them on top of a single shared base model. This is how production systems serve multiple fine-tuned variants without multiplying storage costs.

**Double quantization** compresses the quantization constants themselves, saving an additional ~0.4 bits per parameter. It's enabled by default in the lab config.

---

## Key Terms

| Term | Definition |
|------|-----------|
| Quantization | Representing model weights in fewer bits (e.g., 16-bit → 4-bit) to reduce memory |
| NF4 | NormalFloat 4-bit — a data type designed for normally-distributed neural network weights |
| PTQ | Post-Training Quantization — quantize after training, no retraining needed |
| LoRA | Low-Rank Adaptation — train small rank-decomposed matrices instead of full weight updates |
| QLoRA | Quantized LoRA — 4-bit base model + 16-bit LoRA adapters |
| Rank `r` | The dimension of the LoRA matrices. Typical values: 8, 16, 32. Higher = more capacity. |
| `alpha` | LoRA scaling factor. Usually set to `2×r`. Controls how strongly the adapter influences output. |
| SFTTrainer | Supervised Fine-Tuning Trainer from HuggingFace `trl`. Wraps the training loop. |
| Adapter | The small set of trained LoRA weights saved separately from the base model |
