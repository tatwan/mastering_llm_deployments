# From Quantization to QLoRA, and What Teams Actually Ship

Companion reading for [Lab 4](README.md). Read it before the lab if you want the map first, or after it if you would rather have the numbers in your hands before the explanation. Either way, it answers a question the lab raises but does not stop to settle: **in what order do these techniques happen, and which of them end up in production?**

The short version, so you know where this is going: quantization makes a model small enough to hold, LoRA makes fine-tuning cheap enough to afford, and QLoRA combines them so you can train on a GPU that could not otherwise hold the job. Then, after training, most teams quantize *again*, differently, for serving. The rest of this reading explains why each step exists.

---

## 1. Everything starts with memory

A model is a very large pile of numbers, and every number costs bytes. The first calculation in any deployment conversation is the one you did in Lab 3:

```
memory for the weights = number of parameters × bytes per parameter
```

For a 7-billion-parameter model:

| Precision | Bytes per weight | Weights alone |
|---|---|---|
| FP32 | 4 | 28 GB |
| BF16 / FP16 | 2 | 14 GB |
| INT8 | 1 | 7 GB |
| 4-bit | ~0.5 | ~3.5–4 GB |

That table is only the cost of *holding* the model. Running it adds the KV cache and activations (Lab 3). Training it adds much more, as you will see in section 3.

Now put that next to the hardware. A free Colab T4 has about 15 GB. A 24 GB card (L4, A10G, RTX 4090) is common in the cloud and on desks. An 80 GB H100 is expensive and often hard to get. Most of the decisions in this reading come down to one question: **does the job fit on the GPU you have?**

---

## 2. Quantization: storing each weight in fewer bits

### What it is

Quantization replaces each weight's precise value with the nearest value from a much smaller set. With 4 bits there are only sixteen values available. Each small block of weights (64 in bitsandbytes) gets its own **scale**, a multiplier that stretches those sixteen values to fit that block's range. To use a weight, you look up its level and multiply by the block's scale.

Part A of the lab does this by hand on a real Qwen weight tensor, and three findings from that part carry most of the theory:

- Trained weights cluster tightly around zero. In the tensor we used, 82% of the weights sit within one standard deviation of zero, but the largest one is about 18 standard deviations out.
- Because of that, one scale for the whole tensor is a disaster: the outlier stretches the grid and nearly everything else rounds to the same two or three values. One scale per 64 weights was about 10× more accurate at the same bit width.
- Placing the sixteen levels where the weights actually are, bunched near zero (that is what NF4 does), cut the error by about another 20% for free.

### Why it works at all

It seems like throwing away that much precision should wreck the model. It mostly does not, for two reasons. First, the model's behaviour depends on millions of weights acting together, and small independent rounding errors tend to wash out rather than add up. Second, the schemes that work well (block-wise scales, NF4, and the methods below) are built specifically to protect the few weights that matter most. Larger models tolerate it better than small ones, and 8-bit is generally close to lossless, while 4-bit costs a little quality that you should measure on your own task rather than assume.

### What it buys, and what it costs

Quantization is first a **memory** technique. It lets a model fit on a smaller GPU, or leaves room for more concurrent users, or more KV cache for longer contexts.

Whether it is also a **speed** technique depends on the software doing the math. GPUs do not multiply 4-bit numbers directly in these setups, so every weight has to be unpacked back to 16-bit before each matrix multiply. That unpacking is extra work. On the other hand, generating text one token at a time is usually limited by how fast weights can be read out of GPU memory, not by arithmetic, so fewer bytes to read *can* make generation faster.

Which effect wins depends on the kernels:

- bitsandbytes, which the lab uses, is built for flexibility and memory savings, not speed. Part B measures both sides on the T4: the NF4 model should need a fraction of FP16's memory, and on that hardware it usually generates *more slowly*.
- Serving engines with dedicated 4-bit kernels, such as vLLM running AWQ or GPTQ models with Marlin kernels, or llama.cpp with GGUF files, are built for speed and often do generate faster than 16-bit at small batch sizes.

So "is 4-bit faster?" has no single answer. "Is 4-bit smaller?" is always yes.

### Two moments to quantize

This distinction matters later, so it is worth pinning down now.

The first is on the fly, at load time. bitsandbytes reads the normal 16-bit checkpoint and quantizes it as it loads. There is no preparation step and no extra files, which makes it ideal for experiments and for training (QLoRA). It is not the fastest format to serve.

The second is ahead of time, into a serving format. Methods like GPTQ and AWQ run once, offline, usually with a small set of example text (called *calibration data*) to decide how to round each layer with the least damage. The result is saved as a new checkpoint that serving engines load and run with fast kernels. GGUF, the llama.cpp format, is the same idea for laptops and CPUs. FP8 is an 8-bit floating-point format that recent datacenter GPUs (H100 and newer) run natively.

| Method | When it runs | Typical use |
|---|---|---|
| bitsandbytes NF4 / INT8 | At load time | Experiments, QLoRA training |
| GPTQ | Once, offline, with calibration data | GPU serving (vLLM, TGI) |
| AWQ | Once, offline, with calibration data | GPU serving (vLLM, TGI) |
| GGUF (Q4_K_M, Q5_K_M, Q8_0, ...) | Once, offline | llama.cpp, Ollama, LM Studio on CPUs and laptops |
| FP8 | Once, offline, or at load time | Serving on H100-class GPUs |

All of these are **post-training quantization (PTQ)**: take a finished model and compress it, without retraining.

---

## 3. Fine-tuning, and why the obvious version does not fit

Fine-tuning means continuing to train a pretrained model on your own examples so it behaves the way you need: a house style, a fixed output format, a narrow task, a tone.

The obvious way is **full fine-tuning**: unfreeze every weight and train. Here is what that costs per parameter with the standard setup (mixed precision with the Adam optimizer):

| What has to be in memory | Bytes per parameter |
|---|---|
| The weights, in 16-bit | 2 |
| A gradient for every weight, in 16-bit | 2 |
| A 32-bit master copy of the weights | 4 |
| Adam's two running averages, in 32-bit | 8 |
| **Total** | **~16** |

For the 1.5B model in the lab that is about 24 GB, before a single training example or activation. For a 7B model it is about 112 GB. That is more than one 80 GB H100, for a model whose weights alone fit in 14 GB.

The problem is not the weights. It is everything training needs *alongside* the weights, and that extra cost scales with the number of parameters you train.

That points straight at the fix: train fewer parameters.

A note before going further, because it decides whether you need any of this: fine-tuning changes **how** a model answers much more readily than **what it knows**. If your problem is "the model does not know our documents" or "our facts change weekly," that is retrieval (Lab 6), not fine-tuning. Teams usually try prompting first, then retrieval, and reach for fine-tuning when they need consistent behaviour that prompting cannot hold. Part D of the lab lets you check this yourself: compare the base and tuned answers to the same questions and look at what changed, the style or the facts.

---

## 4. PEFT: train a little, freeze the rest

**Parameter-efficient fine-tuning (PEFT)** is the name for the family of methods that freeze the pretrained model and train only a small number of new or selected parameters. Hugging Face's library for these methods is also called `peft`, which is what the lab imports.

The family includes:

- Adapters: small bottleneck layers inserted between the model's existing layers.
- Prefix tuning and prompt tuning: learned "virtual tokens" prepended to the input, so the model is steered without its weights changing.
- LoRA: small low-rank matrices added alongside existing weight matrices. This is the one that won.

Freezing the base changes the memory math completely. Frozen weights need no gradients and no optimizer state, so they cost only their own storage. Training costs 16 bytes per parameter only for the small set you actually train.

It has other benefits that matter in production, and they are a large part of why PEFT is the default rather than a budget workaround:

- The deliverable is small: you ship tens of megabytes instead of gigabytes.
- One copy of the base model in GPU memory can serve many adapters, one per customer or task, swapped per request.
- The original model is untouched. Because the base is frozen, the fine-tune is less likely to erase general abilities the model already had, and you can always detach the adapter to get the original back.

---

## 5. LoRA: learn the change, not the whole matrix

### The idea

LoRA (Hu et al., 2021) starts from an observation. Take a weight matrix before fine-tuning and after, and subtract them. The difference, ΔW, turns out to have **low rank**: it can be closely approximated by multiplying two thin matrices together.

So LoRA never touches W. It freezes it and learns those two thin matrices, A and B, instead:

```
output = x·W  +  x·A·B
         frozen   trained
```

A takes the input from the layer's full width down to a small rank `r`, and B brings it back up. For a square layer of width `d`, the full matrix has `d × d` numbers and the pair has `2 × d × r`. In the lab, with `d = 1536` and `r = 16`, that is 2.36 million against 49 thousand, about 2%. Across all the linear layers of Qwen2.5-1.5B it came to about 1.2% of the model, roughly 18 million trainable parameters.

The LoRA paper reported cutting trainable parameters by about 10,000× and GPU memory by about 3× compared with full fine-tuning of GPT-3 175B, while matching its quality on the tasks they tested.

### The settings you will actually choose

- Rank `r` is the capacity dial. 8 or 16 is typical for style, tone and format. 32 or 64 when you are teaching more substantial new behaviour. Higher rank means a bigger adapter and slower training, and past a point it buys nothing.
- `lora_alpha` scales the adapter's output by `alpha / r`. Setting it to about twice the rank is the common convention, so that changing `r` does not quietly change how strongly the adapter pushes.
- The target modules decide which layers get adapters. Early practice adapted only the attention projections (`q_proj`, `v_proj`). Adapting every linear layer (`"all-linear"`, as the lab does) is now common because it usually trains better for a modest increase in size.

### Two properties worth remembering

B starts at zero. That means `x·A·B` is exactly zero when training begins, so the model starts identical to the base you already trusted. Part C of the lab checks this inside a real layer. Every change after that is something the optimizer chose.

The adapter can be **merged**. After training, you can compute `W + (alpha/r)·A·B` once, write it back into W, and throw the adapter away. The result is an ordinary model with no extra layers, which any runtime can load. That is Part D's `merge_and_unload()`.

---

## 6. QLoRA: a 4-bit base under 16-bit adapters

### The insight

LoRA shrinks the training overhead to almost nothing, but you still have to hold the frozen base model. For a 7B model in BF16 that is 14 GB, which does not fit on a T4 once activations are added.

QLoRA (Dettmers et al., 2023) asks the obvious next question: if the base is frozen, why store it in 16 bits? Frozen weights are never updated. They are only read during the forward and backward passes. So store them in 4 bits and keep only the adapters, which *are* being updated, in 16-bit.

It is tempting to describe QLoRA as "quantize the model, then fine-tune the quantized model." That is not quite what happens. **The 4-bit weights are never changed.** What trains is the pair of 16-bit adapter matrices sitting next to them.

### What happens during one training step

1. A batch of text goes into the model.
2. At each layer, the frozen 4-bit weights are unpacked to BF16 (this is `bnb_4bit_compute_dtype`), used for the matrix multiply, and discarded.
3. The adapter's contribution `x·A·B` is computed in BF16 and added to the result.
4. The loss is computed at the end.
5. Gradients flow backwards *through* the unpacked base weights (they have to, to reach the adapters in earlier layers), but no gradient is stored for the base and it is never updated.
6. The optimizer updates only A and B.

### What it saves

For a 7B model with a rank-16 adapter on every linear layer (about 40 million trainable parameters):

| Method | Base model | Trainable parts (weights, grads, Adam) | Fits on |
|---|---|---|---|
| Full fine-tuning | 2 bytes × 7B, trained | 16 bytes × 7B ≈ 112 GB total | Several 80 GB GPUs |
| LoRA, 16-bit base | ~14 GB, frozen | ~0.6 GB | A 24 GB GPU |
| QLoRA, 4-bit base | ~4 GB, frozen | ~0.6 GB | A 16 GB T4 |

Activations come on top of every row and grow with batch size and sequence length. That is why the lab uses a batch size of 1 with gradient accumulation, and why `prepare_model_for_kbit_training` turns on gradient checkpointing (recomputing activations during the backward pass instead of storing them).

The QLoRA paper introduced three things together: the **NF4** data type from Part A, **double quantization** (quantizing the block scales too, which saves about 0.4 bits per weight), and **paged optimizers**, which move optimizer state to CPU memory during memory spikes so a long batch does not crash the run. With those, they fine-tuned a 65B model on a single 48 GB GPU and reported results that matched 16-bit fine-tuning on the benchmarks they tested.

### What it costs

Nothing here is free, and knowing the costs is what lets you choose well.

- Steps are slower. Every forward and backward pass unpacks every base weight. That is the same cost Part B measured at inference, now paid twice per step. If LoRA on a 16-bit base fits on your GPU, it will train faster.
- There is a small quality risk. The adapter learns alongside a slightly rounded version of the base. On most tasks the difference from 16-bit LoRA is small, but it is not guaranteed to be zero.
- There is a train/serve mismatch. The adapter was trained to sit on top of the NF4-rounded weights. When you later merge it into the original 16-bit weights (section 7), it sits on a slightly different base from the one it trained with. The effect is usually small, and it is one more reason to evaluate the model you actually ship rather than the checkpoint you trained.

So QLoRA is not "the better LoRA." It is LoRA for when the 16-bit base does not fit.

---

## 7. What teams actually do: the full pipeline

With the pieces in place, here is the pipeline most teams follow. Quantization shows up twice, at two different stages and for two different reasons.

```
1. Train      LoRA on a 16-bit base if it fits, QLoRA on a 4-bit base if it does not
                 ↓
2. Evaluate   the adapter against the base, on held-out examples from your task
                 ↓
3. Decide     serve the adapter as-is, or merge it
                 ↓
4. Merge      fold the adapter into a 16-bit copy of the base
                 ↓
5. Quantize   the merged model again, into the format your serving runtime runs fast
                 ↓
6. Evaluate   again, on the quantized model you are actually going to ship
```

Each step has a reason.

### Train with whatever fits

The training method is a hardware decision. The quantized base used in QLoRA is a training convenience. It lives in memory while you train and is thrown away afterwards; nobody ships it.

### Evaluate before you do anything else

A falling training loss shows only that gradients are flowing. It does not show that the model got better at your task. In the lab, ten examples over three epochs are easy to memorize, so the loss can fall a long way without the model getting any better at questions it has not seen.

### Keep the adapter if you can

If your serving engine supports adapters (vLLM and others can load LoRA adapters and switch between them per request), the adapter itself is the best deliverable. It is small, it can be versioned and rolled back like code, and one base model can serve many of them. This is Part D's first artifact, and the default.

### Merge into 16-bit, never into 4-bit

When the runtime needs a single plain model (llama.cpp, Ollama, many hosted platforms, or when you want to quantize it for serving), merge the adapter into a clean 16-bit copy of the base. Merging into the 4-bit weights means adding precise numbers to rounded ones and rounding again, which stacks one error on top of another. Part D reloads an FP16 base for exactly this reason.

### Quantize again, for serving

The merged model is full size (about 3 GB for the lab's 1.5B, 14 GB for a 7B), and you usually want it smaller for deployment. This time you pick the format for the runtime, not for training:

| Where it will run | Typical format |
|---|---|
| vLLM or TGI on a GPU | AWQ or GPTQ (4-bit), or FP8 on H100-class hardware |
| llama.cpp, Ollama, LM Studio on a laptop or CPU | GGUF, often Q4_K_M or Q5_K_M |
| Plenty of GPU memory, quality is critical | Leave it in BF16 |

This is the `Q4_K_M` step at the end of the lab's GGUF recipe. It is the same idea as NF4, block-wise scales and all, applied by a different tool for a different runtime.

### Evaluate the thing you ship

Quantizing after training is a second source of change, so the numbers you measured in step 2 do not automatically carry over. Run the same evaluation on the quantized model before it goes out. Lab 12 builds this kind of regression check.

### If you have a bigger GPU

The same pipeline works with LoRA in BF16 (or full fine-tuning, if you have lots of data and hardware) in step 1. Everything from step 3 on is identical. This is actually the cleaner route when it fits: no training on rounded weights, no train/serve mismatch, and faster steps. QLoRA's job is to make step 1 possible on hardware that would otherwise refuse it.

### One step further: quantization-aware training

Post-training quantization rounds a model that never expected to be rounded. **Quantization-aware training (QAT)** simulates the rounding during training, so the model learns weights that survive it. It costs more to train and recovers quality that aggressive quantization (4 bits and below) would otherwise lose. Some model releases now ship QAT versions for exactly this reason (Google's Gemma 3 QAT checkpoints are one example), and PyTorch's `torchao` supports it. It is beyond what this course needs, but it is worth knowing the term when you see it on a model card.

---

## 8. Choosing, in practice

When you face this decision on a real project, the questions tend to come in this order:

1. **Do you need to fine-tune at all?** If the model lacks knowledge, use retrieval. If it needs a consistent style, format or narrow skill that prompting cannot hold, fine-tune.
2. **Does a 16-bit base plus LoRA fit on your GPU?** If yes, use LoRA. If no, use QLoRA. Full fine-tuning is for teams with a lot of data, a lot of hardware, and a change too large for an adapter.
3. **Can your runtime serve adapters?** If yes, ship the adapter. If no, merge into a 16-bit base.
4. **Where will it run?** That decides the serving format and whether you quantize again.
5. **Did quality survive?** Measure it on the final artifact, not on the training run.

---

## How this reading maps to the lab

| Idea in this reading | Where you do it in Lab 4 |
|---|---|
| Levels, block-wise scales, NF4, double quantization (section 2) | Part A, cells A1–A7 |
| Memory saved, speed paid, quality kept (section 2) | Part B, cells B1–B8 |
| Why full fine-tuning does not fit; LoRA shapes, zero init, rank (sections 3, 5) | Part C, cells C1–C3 |
| QLoRA: 4-bit frozen base, bf16 adapters, training (section 6) | Part C, cells C4–C11 |
| Adapter as the deliverable, merge into FP16, re-quantize to GGUF (section 7) | Part D, cells D1–D9 |

## Further reading

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021)
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023). NF4, double quantization and paged optimizers all come from here.
- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (2022)
- Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (2023)
- Hugging Face [PEFT documentation](https://huggingface.co/docs/peft), for the other PEFT methods and the adapter-serving options
