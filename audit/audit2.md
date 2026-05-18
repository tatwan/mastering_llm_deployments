Great, I now have a thorough picture of the repo structure vs. the outline. Here's my full audit:

***

## Repo Structure vs. Outline

The repo is organized into 7 folders, but it has **diverged significantly** from the outline — and actually in a *good* way. The outline was written January 2026 and was tightly AWS-centric, while the repo has already evolved beyond it:

| Outline Module | Repo Folder | Gap / Drift |
|---|---|---|
| PyTorch Fundamentals + Inference Lab | `01_Foundations` (implied) | ✅ Present |
| Model Distillation | `03_Model_Optimization/02_knowledge_distillation.ipynb` | ✅ Present, but **uses TF/Keras** — outdated framework framing |
| Model Quantization (INT4/bitsandbytes) | `03_Model_Optimization/04_quantization.ipynb` | ✅ Present |
| Model Pruning | `03_Model_Optimization/03_pruning.ipynb` | ✅ Present |
| Benchmarking | `03_Model_Optimization/05_benchmarking.ipynb` | 🆕 In repo, **not in outline** — great addition |
| Fine-Tuning (LoRA/PEFT/healthcare) | `02_Fine_Tuning/` (4 notebooks) | 🚨 **Not in outline at all** — major scope expansion |
| Docker packaging + AWS ECS | `04_Deployment/03_docker_packaging.md`, `04_aws_ecs_deployment.md` | ⚠️ **Markdown-only**, no executable lab |
| Inference engines (FastAPI, vLLM) | `04_Deployment/01_local_serving.ipynb`, `app.py` | ✅ Present, but **vLLM and TF Serving underrepresented** |
| RAG | `05_RAG/` (LangChain, LlamaIndex, eval) | 🚨 **Explicitly excluded in outline** — complete scope expansion |
| Capstone Hackathon | `Capstone/` | ✅ Present |

 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/233624/5f527b76-c463-44f0-a706-84a2424f1bff/Mastering-LLM-Deployment.md?AWSAccessKeyId=ASIA2F3EMEYERVHMXMCZ&Signature=K9Y3nD0vS9h0XppalPIO7iPaGZ8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEPP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDR6wMmqzs%2BmfBmIRzIISi2pBI2hXkby4hXu198Uf9p6AiALGLwtZKydv9yI95dvMKicO2P2rwYwAqKuH67dpeTcMir8BAi7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMWUfdV0PgUw%2BaQOBuKtAE5yh612JvHVnyoEQZUAtHVrplJdZcV7EUYKnasv3LBBPRg1bNM2NypkRnTRPvBT5MGbADFI6qqdubyPbLgD80laAupNUBAfF%2B0NrPUEmX93SLYmCb%2BFimvzM2BwcV847qG%2BYh3w%2FM%2BMiZkqyCMPreDOJ6bM72g3oIxz5KYX8QjECCQ1LJxjN6l%2BByYmJjS8RcZerK%2B3eQJ8qjq3GkzXur722NodUTB7tVXHvH3D8fmv26vAP%2BWrme4STeYwAyxY6JMEvFS6IHgJUCnjSvD%2Fooc0xx6cFTBUvm9mfqTxUChhagkO%2Bkj1%2BNCK3xmhUWfxfzoy4mR3zz51RKf6F9H7AbOSc9KjNdnpqYn%2FBWuMCaZhjRNZUDPRpiSkG8wwktV92SLgW%2FZBvbDIMLoCh3NFMCTfcoobFin3JISfqZGuqA8LybNVnjyr%2BbHX3y0HRtUunK1E5ujn38Raa%2Fdy6QrdYnoPTmGVhqcph1oNYS8m4HD79oVszbIpOmNBcfDx1IiVg7aSOQvllkzZG5PJ0gG03lVvSWtHeFD7vy92KwxeM0oQd%2FRqo0h%2FTEY2dsRefzRgsbRuEOQu4x1%2F6RMZDmWsRCIu6NvyevEN3kH56UJEKmebNZGfIkjpfVu4nX%2FY1BVIWIJgV3ZZzkWkvhGo6UbMvrgGPFJDXt0l9a74AWh5MT1TGwVMepnLa6xzJUaWWzyJTzhFfMOvETk902fQXe1u4nHyxEb6Jv6q1aNMkP%2BBNWm%2FWnKtcnB17gCxFtzIRV6P9BbdHb8QhskzOJbpkWJj47zTD%2ByavQBjqZAQaqy1YyMOaXwHJyRk%2F0aZ4TYsyfDoMUE9aUZkZQKOwUheauyy4NtA3%2FSmj2tJoV44lPd2PWNlTaSSjjaG5Q0LIEZuTJ6uvT6I%2BjALGWZr0gdR2ueh5V4hlh9h%2Bhbdd6zs56%2FSwjTsp6ZL9Q6WnfA8oj16iZFCTNYUYrwcvk3S86uigOtBK9fmqCKo55%2B3CDlbfYOSxeytiNhQ%3D%3D&Expires=1779099688)

***

## 1. Labs: Add / Replace / Remove

### 🔴 Remove or Demote
- **`02_Fine_Tuning/01_transfer_learning.ipynb`** — Transfer learning with older BERT-style models feels like a step backward. Fine-tuning a full model from scratch is rarely the story anymore; this lab could be replaced with a LoRA/QLoRA intro. 
- **Distillation lab using TensorFlow** — The outline still mentions TensorFlow for distillation, and the repo mirrors this. In May 2026, doing distillation via TF on a BERT/SQuAD setup is anachronistic. Modern distillation is done via Hugging Face `transformers` + PyTorch or via APIs (e.g., OpenAI → student model pipelines). Replace with a **Hugging Face Distilbert → custom student** PyTorch-native approach, or better, an **API-based distillation** lab using a powerful frontier model as teacher. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/233624/5f527b76-c463-44f0-a706-84a2424f1bff/Mastering-LLM-Deployment.md?AWSAccessKeyId=ASIA2F3EMEYERVHMXMCZ&Signature=K9Y3nD0vS9h0XppalPIO7iPaGZ8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEPP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDR6wMmqzs%2BmfBmIRzIISi2pBI2hXkby4hXu198Uf9p6AiALGLwtZKydv9yI95dvMKicO2P2rwYwAqKuH67dpeTcMir8BAi7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMWUfdV0PgUw%2BaQOBuKtAE5yh612JvHVnyoEQZUAtHVrplJdZcV7EUYKnasv3LBBPRg1bNM2NypkRnTRPvBT5MGbADFI6qqdubyPbLgD80laAupNUBAfF%2B0NrPUEmX93SLYmCb%2BFimvzM2BwcV847qG%2BYh3w%2FM%2BMiZkqyCMPreDOJ6bM72g3oIxz5KYX8QjECCQ1LJxjN6l%2BByYmJjS8RcZerK%2B3eQJ8qjq3GkzXur722NodUTB7tVXHvH3D8fmv26vAP%2BWrme4STeYwAyxY6JMEvFS6IHgJUCnjSvD%2Fooc0xx6cFTBUvm9mfqTxUChhagkO%2Bkj1%2BNCK3xmhUWfxfzoy4mR3zz51RKf6F9H7AbOSc9KjNdnpqYn%2FBWuMCaZhjRNZUDPRpiSkG8wwktV92SLgW%2FZBvbDIMLoCh3NFMCTfcoobFin3JISfqZGuqA8LybNVnjyr%2BbHX3y0HRtUunK1E5ujn38Raa%2Fdy6QrdYnoPTmGVhqcph1oNYS8m4HD79oVszbIpOmNBcfDx1IiVg7aSOQvllkzZG5PJ0gG03lVvSWtHeFD7vy92KwxeM0oQd%2FRqo0h%2FTEY2dsRefzRgsbRuEOQu4x1%2F6RMZDmWsRCIu6NvyevEN3kH56UJEKmebNZGfIkjpfVu4nX%2FY1BVIWIJgV3ZZzkWkvhGo6UbMvrgGPFJDXt0l9a74AWh5MT1TGwVMepnLa6xzJUaWWzyJTzhFfMOvETk902fQXe1u4nHyxEb6Jv6q1aNMkP%2BBNWm%2FWnKtcnB17gCxFtzIRV6P9BbdHb8QhskzOJbpkWJj47zTD%2ByavQBjqZAQaqy1YyMOaXwHJyRk%2F0aZ4TYsyfDoMUE9aUZkZQKOwUheauyy4NtA3%2FSmj2tJoV44lPd2PWNlTaSSjjaG5Q0LIEZuTJ6uvT6I%2BjALGWZr0gdR2ueh5V4hlh9h%2Bhbdd6zs56%2FSwjTsp6ZL9Q6WnfA8oj16iZFCTNYUYrwcvk3S86uigOtBK9fmqCKo55%2B3CDlbfYOSxeytiNhQ%3D%3D&Expires=1779099688)
- **`04_Deployment/04_aws_ecs_deployment.md`** (markdown-only) — This is a walkthrough doc, not an executable lab. Given you're moving away from AWS, this is dead weight. 

### 🟡 Replace / Update
- **vLLM lab** — There's no dedicated vLLM notebook. In 2026 vLLM is the de-facto standard for serving open-source LLMs. This desperately needs a hands-on lab, and it runs beautifully on **Hugging Face Spaces** (free GPU) or Colab T4. Replace the `04_aws_ecs_deployment.md` with a **vLLM serving on HF Spaces** lab.
- **Quantization** — Update the quantization notebook to include **GGUF/llama.cpp** (Ollama-style local quantization) alongside bitsandbytes. By 2026, GGUF is universal for local deployment and students will have seen it everywhere.
- **Pruning lab** — Traditional magnitude pruning on SST-2 feels academic. Replace or supplement with a short **MoE sparsity explainer** (Mixtral/Llama-MoE) and show students how to load a sparse model and observe active expert routing. 

### 🟢 Add
- **Ollama local deployment lab** — Dead simple, extremely satisfying for students. Run a quantized model locally in 2 commands, then call it via API. Perfect warm-up lab, works on any laptop, no GPU required.
- **OpenAI-compatible API standardization lab** — vLLM, Ollama, and LM Studio all expose OpenAI-compatible endpoints. A lab showing students how to swap backends (local → cloud) by just changing a base URL is a massive real-world skill in 2026.
- **LLM Gateway / Router pattern** — A short lab on **LiteLLM** to unify calls across providers (OpenAI, Anthropic, local models). This directly replaces the AWS-ECS cost-optimization narrative with something more relevant and platform-agnostic.
- **Basic observability lab** — Add a notebook using **Langfuse** or **Phoenix (Arize)** to trace inference calls. The outline explicitly says no monitoring, but in 2026 this is table stakes and could be a 30-minute lab.

***

## 2. Making It Fun & Engaging

**Setup — ditch the complexity spiral:**
The Colab + HuggingFace combo you used in March is the right call. A clean pattern is:
- **Labs 1-4** (Foundations, Optimization): Google Colab free tier (T4 GPU) with pre-authenticated HF tokens set via `userdata`
- **Labs 5-6** (Serving/Deployment): **Hugging Face Spaces** — students deploy a Gradio app wrapping a quantized model; they get a real public URL in ~5 min. Extremely satisfying and shareable.
- **Capstone**: They can use either environment; judging is based on the public HF Space URL they submit.

**Engagement tactics:**
- **"Break it" challenges** — After each optimization lab, give students a broken/over-pruned or over-quantized model and ask them to diagnose why outputs are garbage. This teaches intuition fast.
- **Live cost calculator** — Build a shared Google Sheet where students fill in their model size, quantization level, and projected load, and it auto-calculates estimated monthly inference cost. Makes cost-optimization visceral.
- **Leaderboard** — For the quantization lab, set up a shared benchmark (same prompt, same dataset, measure tokens/sec + perplexity). Students post their numbers. Instant competitiveness.
- **"Deploy to a friend" moment** — HF Spaces gives a public URL. Have students swap URLs with a partner and try to break each other's deployed model with adversarial prompts. Fun and teaches robustness thinking.

***

## 3. Coverage vs. Outline

The repo actually **over-delivers** on the outline in some areas and has notable gaps in others. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/233624/5f527b76-c463-44f0-a706-84a2424f1bff/Mastering-LLM-Deployment.md?AWSAccessKeyId=ASIA2F3EMEYERVHMXMCZ&Signature=K9Y3nD0vS9h0XppalPIO7iPaGZ8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEPP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDR6wMmqzs%2BmfBmIRzIISi2pBI2hXkby4hXu198Uf9p6AiALGLwtZKydv9yI95dvMKicO2P2rwYwAqKuH67dpeTcMir8BAi7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMWUfdV0PgUw%2BaQOBuKtAE5yh612JvHVnyoEQZUAtHVrplJdZcV7EUYKnasv3LBBPRg1bNM2NypkRnTRPvBT5MGbADFI6qqdubyPbLgD80laAupNUBAfF%2B0NrPUEmX93SLYmCb%2BFimvzM2BwcV847qG%2BYh3w%2FM%2BMiZkqyCMPreDOJ6bM72g3oIxz5KYX8QjECCQ1LJxjN6l%2BByYmJjS8RcZerK%2B3eQJ8qjq3GkzXur722NodUTB7tVXHvH3D8fmv26vAP%2BWrme4STeYwAyxY6JMEvFS6IHgJUCnjSvD%2Fooc0xx6cFTBUvm9mfqTxUChhagkO%2Bkj1%2BNCK3xmhUWfxfzoy4mR3zz51RKf6F9H7AbOSc9KjNdnpqYn%2FBWuMCaZhjRNZUDPRpiSkG8wwktV92SLgW%2FZBvbDIMLoCh3NFMCTfcoobFin3JISfqZGuqA8LybNVnjyr%2BbHX3y0HRtUunK1E5ujn38Raa%2Fdy6QrdYnoPTmGVhqcph1oNYS8m4HD79oVszbIpOmNBcfDx1IiVg7aSOQvllkzZG5PJ0gG03lVvSWtHeFD7vy92KwxeM0oQd%2FRqo0h%2FTEY2dsRefzRgsbRuEOQu4x1%2F6RMZDmWsRCIu6NvyevEN3kH56UJEKmebNZGfIkjpfVu4nX%2FY1BVIWIJgV3ZZzkWkvhGo6UbMvrgGPFJDXt0l9a74AWh5MT1TGwVMepnLa6xzJUaWWzyJTzhFfMOvETk902fQXe1u4nHyxEb6Jv6q1aNMkP%2BBNWm%2FWnKtcnB17gCxFtzIRV6P9BbdHb8QhskzOJbpkWJj47zTD%2ByavQBjqZAQaqy1YyMOaXwHJyRk%2F0aZ4TYsyfDoMUE9aUZkZQKOwUheauyy4NtA3%2FSmj2tJoV44lPd2PWNlTaSSjjaG5Q0LIEZuTJ6uvT6I%2BjALGWZr0gdR2ueh5V4hlh9h%2Bhbdd6zs56%2FSwjTsp6ZL9Q6WnfA8oj16iZFCTNYUYrwcvk3S86uigOtBK9fmqCKo55%2B3CDlbfYOSxeytiNhQ%3D%3D&Expires=1779099688)

**Well-covered ✅**
- The optimization trilogy (distillation, quantization, pruning) has solid lab coverage in `03_Model_Optimization/`
- Fine-tuning went way beyond the outline (4 notebooks including a healthcare-specific one) — good bonus content
- RAG coverage in `05_RAG/` is strong with both LangChain and LlamaIndex, plus a wandb evaluation lab — though it was explicitly "out of scope" in the outline, this is now a gap *in the outline*, not the repo

**Under-covered ⚠️**
- **Inference engines** — vLLM is barely represented; TensorFlow Serving appears in the outline but not meaningfully in the repo; the decision-framework on *when to use which* is underdeveloped
- **Production serving as a complete flow** — The Docker + ECS section is markdown-only with no runnable lab end-to-end; the `01_local_serving.ipynb` is the only executable serving notebook 
- **Benchmarking as a bridge** — `05_benchmarking.ipynb` exists but isn't referenced in the outline at all; it should be more prominently positioned as the *connector* between optimization choices and deployment decisions 

**Outline items that should be dropped from the next version:**
- AWS ECS as the primary deployment target — replace with HF Spaces / Ollama / vLLM local
- TensorFlow Serving — its mindshare in the LLM world has shrunk dramatically; mention it but don't lab it
- SQuAD dataset for distillation — dated; replace with a modern instruction-tuning or chat dataset

----
Here's the full revised outline I've drafted — I'll hold off on pushing to the repo until you approve. Let me walk you through everything:

***

## What's Changing: v1 → v2 at a Glance

| v1 (Jan 2026 Outline) | v2 (May 2026) | Reason |
|---|---|---|
| AWS ECS deployment | **HuggingFace Spaces** | No admin rights, free, public URLs |
| Docker containerization lab | HF Spaces (optional Docker Space) | Reduces complexity |
| TensorFlow Serving | Removed (mention only) | Minimal LLM mindshare in 2026 |
| TF-based distillation | **PyTorch + HF Trainer** | Modern, consistent toolchain |
| No vLLM lab | **Full vLLM module** | Production standard in 2026 |
| No fine-tuning | **QLoRA module** | Inseparable from deployment story now |
| No RAG | **RAG module** | Table stakes for real deployments |
| Local setup required | **100% browser-based** | No admin rights needed |

 [ppl-ai-file-upload.s3.us-east-1.amazonaws](https://ppl-ai-file-upload.s3.us-east-1.amazonaws.com/web/direct-files/attachments/233624/5f527b76-c463-44f0-a706-84a2424f1bff/Mastering-LLM-Deployment.md?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Checksum-Mode=ENABLED&X-Amz-Credential=ASIA2F3EMEYEVXS5ZNFY%2F20260518%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260518T103533Z&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCID0FYVga1FjbHGZRMT3AV5D%2B1BHqLkX%2FWkoX%2FuftNfHOAiEAvxG7vZa%2F1%2BKd8RAK9KGji4aR%2BRbiYj0HrKfshkfK2Gwq9AQIu%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDE0MzX0QiyiXZVSxGSrIBAJ8I64EnQnLXT4MrGWM12HtUz3JQg9RI9jG1ID9UZjfvtwlI4SO%2FdYKPuEcDeN2bvyNmQrz7JmMzO%2FxTW897rN5DY9Yybzxoj744ew5xjg5aVCC9Ta6ZBkpri9CYG0cftgXCYCTUYPRouFDbUWUNrIuy9pfb1DqgEHOjUhq9JMyuYytYo81bEvl5X7JAsEKjVNZX6a%2FB4tzVNu4iH9qYypXfdTzqnRZVFIz%2BPf%2FDVbqjmnURQFSWI2OJLKEUaLAYGMbLPG8aE3nEhKB4COLkMXiKRR1FFce%2BsMad6ZMTLillKW4uJWuSi8xBWfKf9R27dR6moSJ51%2FjSs0WKqfUn1Inj8%2F0YJlg3OxXBOs8kJIWnTs6hm0rUr%2FAdUv5eTGoufUo2JhnzZpX5gdNB%2BqGa2f67nIuB8vGn9%2Fmo3%2FmT0ge5zXLx4%2FFFDOQ%2BGkaDMqk3%2FklaM5dyp470QXZ%2F3lU91jMYrmJfUYkHriASxAJ195WUlCrG2XGMs5lq9HcsLyYE0ETmb7H0OrdDfiFq6JymIZ2sOyzQXna3h%2BNAPnwV9rHhGybbvVcnNIn%2BCGCzCeiKY2rlzi%2BdQyfDtHQ2vOWYb8h2Ioi7mvZ39sosqxQ%2Bk8a0ve%2B81pY9ppl%2FuNbaYgTjh0vXhCR0g9HghGWgxFYHdiaJDQl9oNTSSAjObBHPt8Ao2qrWAOXqWc0tJBSgmgQYZugW%2FF4k6D7RB2XKpbV452yYHmzz8lVD3wryWwtLLwkrHYp1tmcU7uGxrxYBA8LtO8Fu3SJIXyZMKvHq9AGOpgBzZ4t5ZuY46uEPwMqqplBypTXbuxradGUd4fW0n1BBuzQhRHIdnraSHGRLLIf42qQfry2FbQlG9BgjOKLMyEflel2nEDe0GP44qYiUoWo8EHms4mkOp6eKMhOYaehT0eMy%2FT%2BggfYYjgZodzyz1uwEOb%2BUOPL8wLAVTG0%2BP89sdsdFQzeBHbgoDkBZGm6MGc%2FN6hSqfVZ2Bk%3D&X-Amz-SignedHeaders=host&x-id=GetObject&X-Amz-Signature=564ed36f0a7e7a0a53e802ed0e09b0c81cdfb0034c397dad94f1aab99f090277)

***

## Proposed Module Structure

### Day 1 — Foundations, Optimization & Fine-Tuning

**Module 0: Environment Setup & HuggingFace Tour** *(30 min)*
- Setup Colab with GPU, authenticate with `HF_TOKEN` via Colab Secrets (no hardcoded tokens)
- **Lab 0:** Existing `00_huggingface_tour.ipynb` — runs on CPU, no GPU needed 

**Module 1: Transformer Fundamentals & Inference Basics** *(1 hr)*
- Replace `Qwen2.5-0.5B` or `SmolLM2-1.7B` as the demo model (modern, tiny, fast)
- **Lab 1:** Update `01_transformers_basics.ipynb` — load, infer, profile on Colab T4 
- 🔴 **Remove:** `03_bert_gpt.ipynb` and `02_model_architecture.ipynb` — too theoretical for 2 days

**Module 2: Quantization** *(1.5 hrs)*
- Keep `04_quantization.ipynb`, update model to `Mistral-7B` or `Llama-3.2-3B` 
- Add a **shared leaderboard** (Google Sheet) where students post their size/speed/quality numbers

**Module 3: Knowledge Distillation** *(1 hr)*
- Rewrite `02_knowledge_distillation.ipynb` in pure PyTorch — drop TF completely 
- Add a discussion on API-based distillation (frontier model as teacher)

**Module 4: Pruning & Sparsity** *(45 min)*
- Keep `03_pruning.ipynb` for classical magnitude pruning 
- Add Part B: load a quantized MoE model, inspect active expert routing — "this IS sparsity at scale"

**Module 5: QLoRA Fine-Tuning** *(1.5 hrs)*
- Update `04_Fine_Tuning_LLM_Healthcare.ipynb` to use `peft` + `SFTTrainer` + `SmolLM2` 
- Students push their trained LoRA adapter to their own HF Hub profile — **tangible takeaway**

**Module 6: Benchmarking & Decision Framework** *(45 min)*
- Keep `05_benchmarking.ipynb`, add auto-generated comparison chart 
- Students use results to fill in a decision matrix: "which config would you ship?"

***

### Day 2 — Serving, Deployment & Capstone

**Module 7: Inference Engine Landscape** *(45 min, lecture only)*
- FastAPI, vLLM, Ollama, TGI, TF Serving (mention only)
- The OpenAI-compatible endpoint standard — one client, swappable backends

**Module 8: FastAPI Serving** *(1 hr)*
- Update `01_local_serving.ipynb` — build in Colab, deploy to **HF Spaces** 
- Students get a live public URL for their FastAPI LLM service

**Module 9: vLLM Serving** *(1 hr) — NEW*
- New notebook: `04_Deployment/03_vllm_serving.ipynb`
- Run vLLM on Colab T4 → benchmark vs. naive FastAPI → show the "URL swap trick" with OpenAI client

**Module 10: Gradio UI + HF Spaces** *(1 hr)*
- Update `02_gradio_ui.ipynb` — modern Gradio 4.x `gr.ChatInterface` with streaming 
- Deploy to HF Spaces with ZeroGPU; students share URLs and **"attack" each other's bot**

**Module 11: LiteLLM Gateway** *(30 min) — NEW*
- New notebook: `04_Deployment/04_litellm_gateway.ipynb`
- Replaces the AWS cost optimization narrative with a provider-agnostic routing pattern
- Dev on HF API → swap to OpenAI → zero code change

**Module 12: RAG Pipeline** *(1 hr)*
- Update `05_RAG/01_rag_langchain.ipynb` — ChromaDB in-memory (no server), runs on Colab CPU 

**Module 13: Observability Lightning Round** *(20 min, instructor demo)*
- Langfuse tracing added to the Gradio chatbot from Module 10 — live dashboard demo

**Capstone** *(3 hrs)*
- Students pick a theme (customer support bot, code reviewer, medical Q&A, summarizer)
- Must apply one optimization, have a working API, and deploy to HF Spaces
- Judging by: live public URL + README + 2-min demo 

***

## Colab Standard Setup Cell

Every notebook will include this at the top:

```python
import os
from google.colab import userdata

# Students add HF_TOKEN to Colab Secrets tab — never hardcoded
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

!pip install -q transformers datasets accelerate peft trl bitsandbytes
```

Each notebook also gets an **"Open in Colab"** badge linked directly to the GitHub file.

***

