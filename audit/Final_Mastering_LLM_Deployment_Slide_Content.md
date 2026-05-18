# Mastering LLM Deployment

Final consolidated presentation content, May 18, 2026.

This is not a lean syllabus. It is a slide-by-slide content master for a 2-day class where the instructor can teach a smaller live path and keep deep-dive slides as backup/reference. The recommended live path is roughly 95 to 115 slides. The shared deck can include the full 200-slide pack.

## Editorial Judgment

What the research agents agreed on:

- I treated the seven independent research files as the source opinions: `ChatGPT.md`, `DeepSeek.md`, `Qwen.md`, `MinMax.md`, `Kimi.md`, `Perplexity.md`, and `Grok.md`. I treated `Research/Consolidated.md` as the previous consolidation attempt to evaluate against, not as an eighth independent recommendation.
- The class must move from system-level framing to model mechanics to optimization, adaptation, serving, RAG, deployment, observability, and enterprise readiness.
- The most important recurring mental model is "a model is not a product; a deployed LLM is a system."
- The strongest 2026 deployment themes are right-sized models, OpenAI-compatible endpoints, vLLM-style serving, RAG evaluation, gateways/routing, observability, and production governance.
- RAG needs much more depth than the previous slides. Basic RAG is not enough; students need failure modes, chunking, retrieval, reranking, evaluation, security, and observability.
- Labs should be cloud-native, mostly Colab, and built around one progressively richer system.

What I kept from the old slides:

- AI landscape and "tip of the iceberg" level-setting.
- Integrated AI ecosystem framing.
- AI/ML solution dimensions and reality-check slides.
- Track 1 vs Track 2 optimization framing.
- Tokenization and BPE depth.
- RAG as a major expanded section.

What I changed:

- Removed AWS ECS as the main lab path because the new constraint is no local setup and low/free cloud friction.
- Kept Docker/ECS/Kubernetes/private deployment as reference and enterprise-readiness slides, not primary hands-on work.
- Repositioned pruning as a niche/backup topic for LLMs; quantization, SLM selection, batching, caching, and serving engines matter more in practice.
- Made OpenAI-compatible APIs a central serving abstraction, because they let students understand Groq, OpenAI, vLLM, HF Inference Providers, LiteLLM, and custom FastAPI as one interface.
- Added stronger evaluation, observability, security, governance, and decision-record content.

## Current Source Anchors

- vLLM documents OpenAI-compatible serving, PagedAttention, and continuous batching: [vLLM](https://vllm.ai/).
- Groq documents OpenAI-compatible usage with `base_url="https://api.groq.com/openai/v1"` and publishes free-plan rate limits: [Groq OpenAI compatibility](https://console.groq.com/docs/openai), [Groq rate limits](https://console.groq.com/docs/rate-limits).
- Hugging Face documents ZeroGPU for Spaces and Inference Providers with a free tier and OpenAI-compatible client option: [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu), [Inference Providers](https://huggingface.co/docs/hub/models-inference).
- LiteLLM documents a 100+ provider gateway, consistent OpenAI-style formats, retry/fallback logic, budgets, and observability callbacks: [LiteLLM docs](https://docs.litellm.ai/).
- LoRA, QLoRA, PagedAttention, FlashAttention-3, and RAGAS are anchored in the primary papers: [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), [PagedAttention/vLLM](https://papers.cool/arxiv/2309.06180), [FlashAttention-3](https://arxiv.org/abs/2407.08608), [RAGAS](https://arxiv.org/abs/2309.15217).
- Enterprise value realization remains uneven: [McKinsey State of AI 2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai).
- Governance framing should align with NIST AI RMF and the GenAI Profile: [NIST AI RMF GenAI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf).

## Course Rhythm

- Day 1 morning: level-set slowly, build shared vocabulary, inspect models, tokenization, generation controls.
- Day 1 afternoon: accelerate into model sizing, optimization, adaptation, LoRA/QLoRA, and decision frameworks.
- Day 2 morning: serving, APIs, vLLM, scale, then RAG from first principles through evaluation.
- Day 2 afternoon: deployment UI, gateway patterns, observability, enterprise readiness, capstone.

## Slide Legend

- Core: should be in the live deck.
- Backup: strong reference slide, use if students ask or the room needs it.
- Deep dive: technical drill-down, likely not live unless audience is advanced.
- Activity: interactive discussion or mini exercise.
- Handout: keep in shared deck as a printable/checklist reference.

---

# Part 1: Setting the Stage

Goal: establish trust, calibrate the room, and make the demo-to-deployment gap feel real.

### S001 - Welcome: From Demo to Deployment [Core]

- Core message: This class is about deploying LLM systems, not merely calling an API.
- Slide content: "Mastering LLM Deployment: 2 days from model call to working production-pattern system."
- Visual: course map from foundations to capstone.
- Talk track: Students will build the same mental model and then a working stack.
- Lab hook: every major section maps to a notebook or capstone decision.

### S002 - Who Is in the Room? [Core]

- Core message: The class must adapt to beginner, intermediate, and semi-advanced students.
- Slide content: poll: "Used ChatGPT only", "used APIs", "used coding agents", "built RAG", "served models", "production owner".
- Visual: experience ladder.
- Talk track: "No shame in being early. The dangerous gap is having used tools without knowing the system underneath."
- Lab hook: beginners follow the main path; advanced students get stretch goals.

### S003 - The Workshop Contract [Core]

- Core message: This is a workshop with working artifacts, tradeoffs, and debugging.
- Slide content: "You will inspect a model, call a hosted model, quantize, build an API, build RAG, deploy a UI, and present architecture decisions."
- Visual: checklist with completed boxes revealed through the course.
- Talk track: "The output is not confidence. The output is a system you understand."

### S004 - What This Course Is Really About [Core]

- Core message: Deployment is the discipline of making model behavior reliable, useful, affordable, and operable.
- Slide content: four words: reliable, grounded, fast, measurable.
- Visual: notebook demo on left, monitored system on right.
- Talk track: "A demo proves possibility. Deployment proves repeatability."

### S005 - The AI Map [Core]

- Core message: LLMs are one region inside a larger AI landscape.
- Slide content: AI -> ML -> Deep Learning -> GenAI -> LLMs -> LLM applications.
- Visual: nested rings or iceberg.
- Talk track: Reuse the old "tip of the iceberg" idea; LLMs are visible, but data, evaluation, integration, and operations are below water.

### S006 - The Three Eras of AI [Backup]

- Core message: We moved from classification to generation to deployment.
- Slide content: 2012-2018 classify, 2018-2023 generate, 2024-2026 deploy and operate.
- Visual: timeline.
- Talk track: The 2026 problem is not "can it write"; it is "can the system be trusted and afforded at scale."

### S007 - Current AI Reality [Core]

- Core message: Adoption is high, enterprise value is uneven.
- Slide content: McKinsey reports broad AI use but most organizations are still experimenting or piloting; workflow redesign separates high performers.
- Visual: adoption funnel.
- Talk track: Do not weaponize failure stats; use them to explain why engineering discipline matters.
- Citation: McKinsey State of AI 2025.

### S008 - Why AI Pilots Fail [Core]

- Core message: Most failures are system failures, not model failures.
- Slide content: unclear use case, poor data, no integration, no evals, latency/cost surprise, no ownership.
- Visual: failure stack.
- Talk track: "The model rarely fails alone. It fails with friends: data, product, security, ops, and budget."

### S009 - Demo vs Deployment [Core]

- Core message: The demo asks "does it work once"; deployment asks "does it work repeatedly under constraints."
- Slide content: demo: curated prompt, one user, manual judgment. deployment: real users, adversarial inputs, SLA, logs, cost, governance.
- Visual: two-column contrast.
- Lab hook: Lab 4 turns a script into an API; Lab 6 turns RAG into a shareable UI.

### S010 - The Deployment Maturity Ladder [Core]

- Core message: Students need to know where a project currently is.
- Slide content: notebook -> demo -> internal MVP -> production service -> enterprise platform.
- Visual: ladder with exit criteria.
- Talk track: Most teams skip exit criteria and call a demo an MVP.

### S011 - Discriminative vs Generative AI [Core]

- Core message: LLMs behave differently from traditional classifiers.
- Slide content: discriminative answers "what is this"; generative answers "what comes next".
- Visual: classifier boundary vs token probability distribution.
- Talk track: Generative output is open-ended, stochastic, and token-by-token.

### S012 - Why Generative Deployment Is Harder [Core]

- Core message: Generation changes latency, evaluation, memory, and safety.
- Slide content: autoregressive decoding, context windows, hallucination, sampling controls, output constraints.
- Visual: one prompt creating many tokens.
- Lab hook: Lab 2 makes students vary temperature and watch behavior change.

### S013 - The Probability Engine [Backup]

- Core message: An LLM is a next-token probability engine wrapped in product behavior.
- Slide content: text -> tokens -> logits -> sampling -> output.
- Visual: token probability table.
- Talk track: "The magic is the emergent behavior, but deployment starts with this loop."

### S014 - The Integrated AI Ecosystem [Core]

- Core message: LLMs are components inside a larger enterprise system.
- Slide content: UI, API, orchestration, model serving, data, retrieval, tools, observability, governance.
- Visual: layered architecture.
- Talk track: Reuse old integrated ecosystem slides but update with gateways, RAG, evals, and observability.

### S015 - The Five Systems Around the Model [Core]

- Core message: Production needs more than inference.
- Slide content: context system, tool system, safety system, measurement system, cost system.
- Visual: hub-and-spoke model.
- Lab hook: RAG builds context; FastAPI builds service; Gradio builds interface; tracing logs behavior.

### S016 - Organizational Ownership [Backup]

- Core message: LLM deployment is cross-functional.
- Slide content: data owns freshness, ML owns model choices, backend owns API, platform owns infra, security owns controls, product owns acceptance.
- Visual: RACI-style table.
- Talk track: Architecture is the contract between teams.

### S017 - Course Journey [Core]

- Core message: The course intentionally crawls, walks, runs, then flies.
- Slide content: Part 1-3 crawl, Part 4-6 walk/run, Part 7-8 run, Part 9 synthesize.
- Visual: road map.

### S018 - Activity: Demo or Deployment? [Activity]

- Core message: Students can classify maturity from symptoms.
- Slide content: five scenarios; students say "demo", "MVP", or "production".
- Visual: decision cards.
- Talk track: Use this to surface assumptions in the room.

### S019 - Part 1 Takeaways [Core]

- Core message: The model is not the product; the deployed system is the product.
- Slide content: 1. LLMs sit inside AI. 2. Pilots fail around the model. 3. Generation changes operations. 4. Systems thinking wins.
- Visual: four-tile recap.

---

# Part 2: The Modern GenAI Stack

Goal: orient students around the tools and reduce tool-sprawl anxiety.

### S020 - The GenAI Stack at a Glance [Core]

- Core message: Every tool belongs to a layer.
- Slide content: framework, model hub, inference, orchestration, application, observability.
- Visual: stack.
- Talk track: "When confused, ask which layer you are debugging."

### S021 - Layer 1: Frameworks [Core]

- Core message: PyTorch is the default language of open LLM implementation.
- Slide content: PyTorch, TensorFlow, JAX; why PyTorch dominates LLM libraries.
- Visual: framework layer with PyTorch highlighted.
- Talk track: TensorFlow knowledge transfers, but this course uses PyTorch/Hugging Face.

### S022 - TensorFlow vs PyTorch: Practical Rule [Backup]

- Core message: Use the framework the model ecosystem uses.
- Slide content: LLM loading, PEFT, vLLM, bitsandbytes, transformers primarily follow PyTorch patterns.
- Visual: decision rule.
- Talk track: This is not framework religion; it is ecosystem fit.

### S023 - Layer 2: Model Hubs [Core]

- Core message: Hugging Face is infrastructure, not only a website.
- Slide content: models, datasets, spaces, model cards, inference providers.
- Visual: "GitHub + npm + Docker Hub for models".
- Lab hook: Lab 1 reads model cards and loads a small model.

### S024 - Model Cards as Deployment Contracts [Core]

- Core message: A model card tells you whether a model is deployable for your use case.
- Slide content: license, intended use, architecture, context, quantization, evals, safety, limitations.
- Visual: annotated model card.
- Talk track: "Ignore the model card and you inherit unknown risk."

### S025 - Layer 3: Inference Providers [Core]

- Core message: Hosted inference can be a fast, low-friction lab and prototype path.
- Slide content: Groq, Hugging Face Inference Providers, Together, OpenAI, Azure, Bedrock, Vertex.
- Visual: provider grid.
- Lab hook: Groq is the default lab backend because it is OpenAI-compatible and low-friction.

### S026 - Layer 4: Orchestration [Core]

- Core message: Orchestration connects models to data, tools, workflows, and state.
- Slide content: LangChain, LangGraph, LlamaIndex, Semantic Kernel, raw Python.
- Visual: model connected to retrieval and tools.
- Talk track: Frameworks speed you up until they hide the bug. Know the escape hatch.

### S027 - LangChain vs LlamaIndex [Core]

- Core message: Pick by problem shape.
- Slide content: LangChain/LangGraph for stateful workflows and tools; LlamaIndex for data-heavy RAG; raw Python for simple pipelines.
- Visual: decision tree.
- Lab hook: Labs use mostly raw Python plus light LangChain to reveal the mechanics.

### S028 - The Abstraction Tradeoff [Backup]

- Core message: Abstractions trade speed for control and debuggability.
- Slide content: raw SDK, helper library, orchestration framework, managed platform.
- Visual: control vs speed curve.

### S029 - Deployment Archetypes [Core]

- Core message: There are different paths to ship depending on maturity.
- Slide content: demo UI, API-first service, managed endpoint, private serving, gateway platform.
- Visual: progression.
- Talk track: Do not overbuild Phase 1; do not underbuild Phase 4.

### S030 - Gradio: Fast Feedback [Core]

- Core message: Gradio is for human feedback and shareable demos.
- Slide content: chat interface, sliders, examples, share links.
- Visual: UI mock.
- Lab hook: Lab 6 turns RAG into a visible app.

### S031 - FastAPI: API-First Control [Core]

- Core message: FastAPI is useful when you need custom logic around model calls.
- Slide content: validation, auth, streaming, logging, business logic, OpenAPI docs.
- Visual: request path through FastAPI.
- Lab hook: Lab 4 builds an OpenAI-compatible FastAPI wrapper.

### S032 - Hugging Face Spaces [Core]

- Core message: Spaces are a low-friction way to showcase and persist demos.
- Slide content: `app.py`, `requirements.txt`, secrets, hardware, ZeroGPU.
- Visual: git push to live app.
- Citation: HF ZeroGPU and Spaces docs.

### S033 - The 4-Phase Deployment Pipeline [Core]

- Core message: Local -> prototype -> cloud MVP -> production scale.
- Slide content: stage, goal, tools, exit criteria.
- Visual: pipeline.
- Talk track: This becomes the course map.

### S034 - Phase Exit Criteria [Handout]

- Core message: You should know when to move to the next phase.
- Slide content: Phase 1 runs; Phase 2 gets feedback; Phase 3 has auth/logs/evals; Phase 4 has SLOs, governance, cost controls.
- Visual: checklist.

### S035 - OpenAI-Compatible Endpoints Preview [Core]

- Core message: One client can call many backends by changing `base_url`.
- Slide content: same client for OpenAI, Groq, vLLM, HF router, LiteLLM, custom FastAPI.
- Visual: one application, many endpoints.
- Talk track: This is the single most useful serving abstraction in the course.

### S036 - Activity: Place the Tool in the Stack [Activity]

- Core message: Students learn tool taxonomy by sorting.
- Slide content: PyTorch, transformers, PEFT, vLLM, FastAPI, Gradio, Chroma, Langfuse, LiteLLM, RAGAS.
- Visual: blank stack.

### S037 - Part 2 Takeaways [Core]

- Core message: The stack is understandable when layered.
- Slide content: 1. PyTorch/HF are the open model path. 2. Orchestration is optional but useful. 3. Deployment maturity determines tooling. 4. OpenAI compatibility decouples apps from backends.

---

# Part 3: Understanding Language Models

Goal: teach only the internals needed for deployment decisions.

### S038 - Why Internals Matter [Core]

- Core message: You do not need to train transformers from scratch, but you need to understand what constrains them.
- Slide content: tokens, attention, memory, sampling, context, KV cache.
- Visual: deployment constraints mapped to model internals.

### S039 - Text to Tokens [Core]

- Core message: Tokens are the currency of cost, latency, and context.
- Slide content: text -> tokenizer -> token IDs -> model -> decoded output.
- Visual: tokenized sentence.
- Lab hook: Lab 1 unwraps the pipeline.

### S040 - BPE and Subword Tokenization [Core]

- Core message: Tokenizers split text into frequent subword pieces, not words.
- Slide content: BPE intuition, vocabulary, token IDs, special tokens.
- Visual: word split examples.
- Talk track: This honors the old Part 3B tokenization depth.

### S041 - Tokens Are Not Equal Across Languages [Backup]

- Core message: Tokenization affects multilingual cost and fairness.
- Slide content: English vs Arabic vs code vs emoji examples.
- Visual: token count comparison.
- Talk track: Always measure token counts with the target tokenizer.

### S042 - Context Window [Core]

- Core message: Context is working memory, not permanent knowledge.
- Slide content: system prompt + history + retrieved docs + user input + output budget.
- Visual: context budget bar.
- Lab hook: Lab 2 tracks growing conversation history.

### S043 - Context Overflow Is a Production Bug [Core]

- Core message: Long conversations and RAG can silently degrade.
- Slide content: truncation, lost instructions, missing citations, higher latency, higher cost.
- Visual: overflowing context bar.

### S044 - Transformer in 90 Seconds [Core]

- Core message: Attention lets tokens condition on other tokens.
- Slide content: embeddings, attention, feed-forward layers, residual stack.
- Visual: simple transformer block.
- Talk track: Keep math light; connect immediately to memory and latency.

### S045 - Encoder, Decoder, Encoder-Decoder [Backup]

- Core message: Architecture type affects use case.
- Slide content: encoder for embeddings/classification, decoder for generation, encoder-decoder for translation/summarization.
- Visual: three model shapes.

### S046 - Base, Instruction, Chat, Reasoning Models [Core]

- Core message: Model tuning determines the interaction contract.
- Slide content: base continues text; instruction follows tasks; chat uses roles; reasoning models spend more compute on deliberation.
- Visual: same prompt through four model types.

### S047 - Chat Templates [Core]

- Core message: Instruction models expect special formatting.
- Slide content: role messages become model-specific tokens.
- Visual: raw prompt vs formatted chat template.
- Lab hook: Lab 2 shows raw vs template output.

### S048 - Loading a Model [Core]

- Core message: Loading has three required pieces: model weights, tokenizer, config.
- Slide content: `AutoTokenizer`, `AutoModelForCausalLM`, revision pinning, dtype, device map.
- Visual: loading checklist.

### S049 - Device and Memory Basics [Core]

- Core message: Parameter count and precision determine baseline memory.
- Slide content: FP32 4 bytes/param, FP16/BF16 2 bytes/param, INT8 1 byte, INT4 0.5 byte.
- Visual: memory ladder.
- Lab hook: Lab 1 and Lab 3 compute model memory.

### S050 - KV Cache [Core]

- Core message: Generation gets expensive because previous key/value states must stay available.
- Slide content: longer context and more concurrent users grow KV cache memory.
- Visual: per-token cache accumulating.
- Talk track: This sets up PagedAttention and vLLM.

### S051 - Prefill vs Decode [Deep dive]

- Core message: LLM latency has two phases.
- Slide content: prefill processes prompt; decode generates one token at a time.
- Visual: timeline with TTFT and tokens/sec.
- Talk track: Optimization differs by phase.

### S052 - Generation Is Controlled Randomness [Core]

- Core message: Sampling settings are deployment controls.
- Slide content: temperature, top-p, top-k, max tokens, stop sequences, penalties.
- Visual: control panel.
- Lab hook: Lab 2 varies decoding settings.

### S053 - Temperature [Core]

- Core message: Lower temperature improves consistency; higher temperature increases variety.
- Slide content: extraction/code: low; brainstorming: medium; creative: higher.
- Visual: probability distribution flattening.

### S054 - Top-p and Top-k [Backup]

- Core message: Nucleus sampling limits the candidate set.
- Slide content: top-p keeps smallest token set crossing cumulative probability p; top-k keeps fixed count.
- Visual: sorted probability bars.

### S055 - Stop Sequences and Max Tokens [Core]

- Core message: Output limits are product controls and cost controls.
- Slide content: use stop tokens, max output, JSON boundaries, streaming cancel.
- Visual: generated output with stop marker.

### S056 - Structured Output [Core]

- Core message: Production systems prefer parseable outputs over free-form prose.
- Slide content: JSON mode, schema validation, retries, constrained decoding.
- Visual: bad prose output vs valid JSON object.
- Lab hook: optional stretch in Lab 4.

### S057 - Small Language Model Revolution [Core]

- Core message: Right-sized models often beat giant models for deployment.
- Slide content: cheaper, faster, private, lower latency, easier fine-tuning; route hard cases to larger models.
- Visual: model routing pyramid.
- Talk track: Avoid "bigger is always better."

### S058 - Model Routing [Core]

- Core message: Modern systems often use multiple models.
- Slide content: simple query -> small model; hard query -> frontier model; private data -> private model; formatting -> fine-tuned small model.
- Visual: router.
- Lab hook: Lab 4 compares multiple backends with same client.

### S059 - Cost Equation [Core]

- Core message: Cost is tokens times model price plus retries, embeddings, reranking, infrastructure, and idle time.
- Slide content: cost/request = input + output + retrieval + judge + retries + serving overhead.
- Visual: stacked cost bar.

### S060 - Beginner Misconceptions [Backup]

- Core message: Correct the myths before they become design mistakes.
- Slide content: "context is memory", "RAG fixes hallucinations automatically", "fine-tuning adds facts", "temperature 0 makes truth", "bigger is always better".
- Visual: myth vs reality.

### S061 - Activity: Diagnose the Constraint [Activity]

- Core message: Map symptoms to root causes.
- Slide content: slow first token, repeated output, hallucinated policy, rate limit, OOM, high bill.
- Visual: symptom cards.

### S062 - Part 3 Takeaways [Core]

- Core message: Deployment decisions are easier once tokens, memory, generation, and context are visible.
- Slide content: tokens drive cost; KV cache drives serving memory; sampling drives behavior; right-sized models often win.

---

# Part 4: Making Models Smaller and Faster

Goal: teach computational optimization without pretending every technique is equally practical.

### S063 - The Deployment Bottleneck [Core]

- Core message: Many deployment problems are not quality problems; they are latency, memory, and cost problems.
- Slide content: model too large, slow first token, low throughput, high GPU bill.
- Visual: bottleneck dashboard.

### S064 - Two Optimization Tracks [Core]

- Core message: Separate functional quality from computational efficiency.
- Slide content: Track 1 changes behavior: prompting, RAG, fine-tuning. Track 2 changes efficiency: quantization, distillation, pruning/sparsity, serving optimization.
- Visual: two tracks.
- Talk track: Keep this old-slide concept. It is excellent.

### S065 - Do Not Mix the Diagnosis [Core]

- Core message: Use the right tool for the failing constraint.
- Slide content: hallucination -> RAG/eval; slow throughput -> serving/batching; VRAM OOM -> quantize/smaller model; style mismatch -> prompt/fine-tune.
- Visual: symptom-to-lever table.

### S066 - Optimization Decision Matrix [Core]

- Core message: The first practical lever is usually not the most exotic one.
- Slide content: try smaller model, quantize, improve serving, cache, route, then distill/fine-tune if needed.
- Visual: one-page matrix.

### S067 - Practical Optimization Order [Core]

- Core message: Start cheap and reversible.
- Slide content: measure baseline -> right-size model -> quantize -> batch/cache -> vLLM/TGI/SGLang -> evaluate -> consider distillation.
- Visual: staircase.

### S068 - Quantization: The Workhorse [Core]

- Core message: Quantization reduces memory by storing weights at lower precision.
- Slide content: FP32, FP16/BF16, INT8, INT4/NF4.
- Visual: precision ladder.
- Lab hook: Lab 3 benchmarks FP16 vs NF4.

### S069 - Why Quantization Works [Backup]

- Core message: LLM weights are robust to some precision reduction.
- Slide content: ranges, scale factors, per-channel/per-group quantization.
- Visual: continuous values snapped to bins.

### S070 - PTQ vs QAT [Core]

- Core message: Post-training quantization is practical; quantization-aware training is heavier.
- Slide content: PTQ after training; QAT simulates quantization during training; use QAT only when quality loss matters and you can retrain.
- Visual: forked workflow.

### S071 - NF4 and QLoRA Connection [Core]

- Core message: NF4 is central to memory-efficient fine-tuning.
- Slide content: NormalFloat4, double quantization, paged optimizers.
- Citation: QLoRA paper.
- Lab hook: Lab 3 uses bitsandbytes NF4.

### S072 - AWQ, GPTQ, GGUF [Backup]

- Core message: Different quantization formats serve different runtimes.
- Slide content: AWQ/GPTQ for GPU inference; GGUF for llama.cpp/Ollama CPU/local; bitsandbytes for training/lab workflows.
- Visual: format to runtime map.

### S073 - Quantization Is Not Free [Core]

- Core message: Measure quality, speed, memory, and compatibility.
- Slide content: possible quality loss, kernel support, dequant overhead, model-specific results.
- Visual: tradeoff triangle.

### S074 - FlashAttention [Deep dive]

- Core message: FlashAttention is a lossless attention optimization, not weight compression.
- Slide content: reduces memory reads/writes; FlashAttention-3 uses Hopper capabilities and FP8 paths.
- Citation: FlashAttention-3 paper.

### S075 - Distillation: Teaching a Smaller Model [Core]

- Core message: Distillation transfers behavior from teacher to student.
- Slide content: teacher outputs, soft labels, temperature, student training.
- Visual: teacher-student diagram.

### S076 - When Distillation Is Worth It [Core]

- Core message: Distill only when off-the-shelf small models are insufficient.
- Slide content: strict edge constraints, high request volume, proprietary teacher behavior, repeated narrow task.
- Visual: decision rule.

### S077 - Distillation Reality in 2026 [Backup]

- Core message: Many strong small models are already distilled or heavily instruction-tuned.
- Slide content: evaluate existing SLMs before building a distillation pipeline.
- Talk track: The cheapest distillation is the one the model provider already did.

### S078 - Pruning Basics [Backup]

- Core message: Pruning removes weights or structures but often underdelivers for transformer serving.
- Slide content: unstructured vs structured pruning; sparse hardware requirement.
- Visual: sparse matrix vs dense smaller matrix.

### S079 - Modern Sparsity and MoE [Core]

- Core message: Mixture of Experts is the practical sparsity story, but it has memory tradeoffs.
- Slide content: route token to a subset of experts; faster active compute but many weights may need to be resident.
- Visual: router selecting experts.

### S080 - Serving Optimization Preview [Core]

- Core message: Runtime optimization often beats model surgery.
- Slide content: batching, PagedAttention, prefix caching, speculative decoding, streaming, request cancellation.
- Visual: GPU utilization before/after.

### S081 - Evaluation Travels with Optimization [Core]

- Core message: Optimization without evals is guessing.
- Slide content: compare memory, latency, throughput, cost, task score, hallucination rate.
- Visual: before/after scorecard.

### S082 - Activity: Pick the Optimization [Activity]

- Core message: Students practice constraint-driven thinking.
- Slide content: 8 GB GPU, 200 ms latency, bad JSON, changing docs, 100K requests/day, privacy constraint.
- Visual: scenario cards.

### S083 - Part 4 Takeaways [Core]

- Core message: Make models smaller only when measurement says that is the bottleneck.
- Slide content: quantization first, distillation selectively, pruning cautiously, runtime optimization constantly.

---

# Part 5: Adapting Models to Your Domain

Goal: teach how to improve usefulness without jumping prematurely to fine-tuning.

### S084 - The Customization Problem [Core]

- Core message: A base model is a generalist; your system must supply task, knowledge, format, and policy.
- Slide content: missing instructions, missing facts, wrong style, invalid output, weak domain reasoning.
- Visual: four adaptation needs.

### S085 - Prompting, RAG, Fine-Tuning [Core]

- Core message: These are different levers, not interchangeable magic.
- Slide content: prompting guides behavior at runtime; RAG supplies fresh facts; fine-tuning changes reusable behavior.
- Visual: adaptation spectrum.

### S086 - Adaptation Decision Tree [Core]

- Core message: Start with the least permanent lever.
- Slide content: can prompt solve it? does knowledge change? is behavior repeated and stable? can you evaluate it?
- Visual: decision tree.

### S087 - Prompting Is Runtime Programming [Core]

- Core message: Prompts are part of the application contract.
- Slide content: role, task, context, constraints, examples, output schema.
- Visual: prompt anatomy.

### S088 - Prompting for Deployment [Core]

- Core message: Production prompts need versioning and tests.
- Slide content: prompt ID, model version, eval set, rollback, change log.
- Visual: prompt lifecycle.

### S089 - Few-Shot Examples [Backup]

- Core message: Few-shot prompts teach format and decision boundaries.
- Slide content: examples as tests; keep examples representative and compact.
- Visual: prompt with examples.

### S090 - Chain-of-Thought Caution [Backup]

- Core message: Do not expose hidden reasoning as a product contract.
- Slide content: ask for concise rationale or structured answer; use private reasoning controls where provider supports them.
- Visual: hidden scratchpad vs final answer.

### S091 - RAG vs Fine-Tuning [Core]

- Core message: RAG is for knowledge; fine-tuning is for behavior.
- Slide content: dynamic docs, citations, policy updates -> RAG. style, format, repeated task -> fine-tune.
- Visual: open book vs trained habit.

### S092 - RAG and Fine-Tuning Together [Core]

- Core message: Many strong systems combine both.
- Slide content: fine-tuned model for domain format plus RAG for current facts.
- Visual: base + adapter + retrieval.

### S093 - Full Fine-Tuning Problem [Core]

- Core message: Full fine-tuning is expensive to train, store, deploy, and govern.
- Slide content: new model copy per task, VRAM, catastrophic forgetting risk, rollout complexity.
- Visual: one giant per task.

### S094 - PEFT [Core]

- Core message: Parameter-efficient fine-tuning updates a small set of parameters.
- Slide content: freeze base, train adapter; lower memory; easier storage and task swapping.
- Visual: frozen base with small adapter.

### S095 - LoRA Mechanics [Core]

- Core message: LoRA learns a low-rank update to selected weight matrices.
- Slide content: W_new = W + BA, rank r, alpha, target modules.
- Citation: LoRA paper.
- Lab hook: Lab 3 attaches LoRA adapters.

### S096 - Choosing LoRA Rank [Backup]

- Core message: Rank controls adapter capacity and overfitting risk.
- Slide content: r=4 simple style, r=8-16 common starting point, r=32+ harder tasks.
- Visual: capacity slider.

### S097 - QLoRA [Core]

- Core message: QLoRA trains adapters through a 4-bit frozen base.
- Slide content: NF4 base, 16-bit adapter updates, double quantization, paged optimizers.
- Citation: QLoRA paper.

### S098 - Adapter Deployment Patterns [Core]

- Core message: You can serve one base model with many task adapters.
- Slide content: load adapter per tenant/task; merge for speed; keep separate for flexibility.
- Visual: one base, many adapters.

### S099 - Fine-Tuning Readiness Checklist [Handout]

- Core message: Do not fine-tune until you can evaluate.
- Slide content: stable task, high-quality examples, held-out eval, baseline, failure categories, rollback plan.
- Visual: checklist.

### S100 - Synthetic Data and Distillation [Backup]

- Core message: Synthetic examples can help but amplify teacher bias and mistakes.
- Slide content: generate, filter, deduplicate, human spot-check, eval.
- Visual: data flywheel.

### S101 - Preference Tuning Awareness [Backup]

- Core message: DPO/RLHF-style tuning exists but is not a first lab topic.
- Slide content: use when ranking preferences matter and you have preference data.
- Visual: chosen vs rejected examples.

### S102 - Activity: Prompt, RAG, or Fine-Tune? [Activity]

- Core message: Students classify adaptation levers.
- Slide content: support policy, JSON extraction, brand tone, new product catalog, legal research, SQL style.
- Visual: three buckets.

### S103 - Part 5 Takeaways [Core]

- Core message: Adaptation is a ladder, not a reflex.
- Slide content: prompt first, RAG for changing knowledge, PEFT/LoRA for stable behavior, QLoRA for low-cost open-model tuning.

---

# Part 6: Serving Models

Goal: teach the path from model call to reliable endpoint.

### S104 - From Model File to Service [Core]

- Core message: Serving means exposing inference behind an API contract.
- Slide content: request, validation, queue, model, stream, response, logs.
- Visual: API path.

### S105 - Inference Engine Landscape [Core]

- Core message: Different engines optimize different constraints.
- Slide content: FastAPI, vLLM, TGI, SGLang, Ollama, managed APIs.
- Visual: comparison matrix.

### S106 - FastAPI Use Case [Core]

- Core message: Use FastAPI when control and application logic matter.
- Slide content: auth, custom routing, validation, business rules, streaming, docs.
- Visual: wrapper around backend model.

### S107 - Ollama and Local Dev [Backup]

- Core message: Ollama is excellent for local development and demos, not a default high-throughput production engine.
- Slide content: GGUF, easy install, local privacy, limited concurrency.
- Visual: laptop dev loop.

### S108 - vLLM [Core]

- Core message: vLLM is a production-grade serving engine for high-throughput open-model inference.
- Slide content: OpenAI-compatible API, PagedAttention, continuous batching, quantization, metrics.
- Citation: vLLM docs and PagedAttention paper.

### S109 - PagedAttention [Core]

- Core message: PagedAttention manages KV cache like virtual memory pages.
- Slide content: reduce fragmentation, share blocks, serve more concurrent requests.
- Visual: contiguous vs paged KV cache.

### S110 - Continuous Batching [Core]

- Core message: New requests can join ongoing generation work.
- Slide content: static batching waits; continuous batching keeps GPU busy.
- Visual: request timeline.

### S111 - TGI and SGLang [Backup]

- Core message: vLLM is not the only serious engine.
- Slide content: TGI for HF enterprise path; SGLang for structured generation and prefix/cache-heavy workloads.
- Visual: engine map.

### S112 - The OpenAI-Compatible Standard [Core]

- Core message: Standardizing the API decouples application code from serving backend.
- Slide content: change `base_url` and model name, keep client code.
- Visual: one OpenAI client to Groq, vLLM, HF router, LiteLLM, custom FastAPI.
- Citation: Groq, HF, vLLM docs.

### S113 - Building an OpenAI-Compatible Wrapper [Core]

- Core message: You can create your own endpoint that existing SDKs can call.
- Slide content: `/v1/models`, `/v1/chat/completions`, streaming SSE chunks.
- Visual: request/response schema.
- Lab hook: Lab 4.

### S114 - Streaming [Core]

- Core message: Streaming improves perceived latency and user experience.
- Slide content: time to first token, tokens per second, cancel button.
- Visual: token stream.

### S115 - Latency Metrics [Core]

- Core message: "Latency" is not one metric.
- Slide content: TTFT, tokens/sec, end-to-end latency, queue time, p50/p95/p99.
- Visual: latency timeline.

### S116 - Throughput Metrics [Core]

- Core message: Serving is about users and tokens, not one prompt.
- Slide content: requests/sec, tokens/sec, concurrent sessions, GPU utilization, queue depth.
- Visual: dashboard.

### S117 - Rate Limits and Quotas [Core]

- Core message: Hosted APIs fail gracefully only if you design for limits.
- Slide content: RPM, TPM, RPD, retries, backoff, fallbacks, headers.
- Citation: Groq rate-limit docs.

### S118 - Request Cancellation [Backup]

- Core message: Cancel wasted generation when users leave or requests time out.
- Slide content: streaming cancel, timeout, queue removal, cost savings.
- Visual: canceled request.

### S119 - Caching [Core]

- Core message: Cache repeated work when safe.
- Slide content: prompt-prefix caching, semantic cache, response cache, retrieval cache.
- Visual: cache layers.

### S120 - Speculative Decoding [Deep dive]

- Core message: Draft models can speed decoding when supported.
- Slide content: small draft model proposes tokens; large model verifies.
- Visual: draft/verify loop.

### S121 - Deployment Topologies [Core]

- Core message: Architecture changes with scale and privacy.
- Slide content: managed API, self-hosted vLLM, hybrid gateway, private VPC, edge SLM.
- Visual: topology cards.

### S122 - Docker and Containers [Backup]

- Core message: Containers matter for reproducibility, even if labs avoid local setup.
- Slide content: image size, CUDA versions, model cache, healthcheck, secrets.
- Visual: container stack.

### S123 - Kubernetes/ECS Awareness [Backup]

- Core message: Enterprise deployment often uses orchestration, but this class will not require it hands-on.
- Slide content: autoscaling, GPU nodes, rolling deploy, cold starts, cost.
- Visual: cluster.

### S124 - Serving Anti-Patterns [Core]

- Core message: Small mistakes become expensive at scale.
- Slide content: load model per request, no streaming, no backoff, no version logs, one giant model for all traffic, no eval gate.
- Visual: warning list.

### S125 - Activity: Choose the Serving Pattern [Activity]

- Core message: Serving choices follow workload shape.
- Slide content: internal chatbot, 1000 RPS API, sensitive documents, hackathon demo, edge app.
- Visual: scenario cards.

### S126 - Part 6 Takeaways [Core]

- Core message: Serving turns model behavior into an API product.
- Slide content: OpenAI compatibility, vLLM for throughput, FastAPI for control, streaming for UX, measurement for operations.

---

# Part 7: RAG - Grounding Models in Your Data

Goal: expand beyond basic RAG into production RAG thinking.

### S127 - Why RAG Exists [Core]

- Core message: Models need external, current, permissioned knowledge.
- Slide content: private docs, changing facts, citations, lower retraining cost.
- Visual: model with open book.

### S128 - RAG Is a Pipeline [Core]

- Core message: RAG is not "add a vector database."
- Slide content: ingest, parse, chunk, embed, index, retrieve, rerank, prompt, generate, evaluate.
- Visual: pipeline.

### S129 - The RAG Iceberg [Core]

- Core message: The API call is the visible tip; quality lives below the waterline.
- Slide content: chunking, metadata, embeddings, retrieval, reranking, prompt, eval, permissions, monitoring.
- Visual: iceberg.

### S130 - Four-Stage RAG Pipeline [Core]

- Core message: Ingest -> index -> retrieve -> generate.
- Slide content: stage goals and outputs.
- Visual: four-stage flow.
- Lab hook: Lab 5 builds this.

### S131 - RAG Failure Modes by Stage [Core]

- Core message: Debug RAG by locating the failing stage.
- Slide content: bad parsing, bad chunks, weak embeddings, missing metadata, wrong top-k, prompt ignores context, judge missing.
- Visual: failure map.

### S132 - Ingestion [Core]

- Core message: Garbage-in RAG is common and quiet.
- Slide content: PDFs, HTML, tables, OCR, duplicates, freshness, source IDs.
- Visual: messy document to clean text.

### S133 - Chunking Is the Highest-Impact Decision [Core]

- Core message: Chunk boundaries shape retrieval quality.
- Slide content: fixed, recursive, semantic, markdown-aware, parent-child.
- Visual: document sliced different ways.

### S134 - Chunk Size Tradeoff [Core]

- Core message: Small chunks improve precision; large chunks preserve context.
- Slide content: 200-500 tokens as starting range; tune by eval.
- Visual: precision/recall curve.

### S135 - Chunking by Document Type [Backup]

- Core message: Policies, manuals, code, tables, chats, and tickets require different chunking.
- Slide content: structure-aware strategies.
- Visual: document-type table.

### S136 - Embeddings [Core]

- Core message: Embeddings map text into semantic vector space.
- Slide content: query and chunks become vectors; similarity finds candidates.
- Visual: 2D vector clusters.
- Lab hook: Lab 5 plots embeddings.

### S137 - Embedding Model Selection [Core]

- Core message: Pick embeddings by domain, language, cost, latency, and privacy.
- Slide content: local MiniLM/BGE for labs; paid/provider embeddings for production quality where justified.
- Visual: embedding choice matrix.

### S138 - Similarity Search [Core]

- Core message: Vector search approximates nearest neighbors at scale.
- Slide content: cosine similarity, dot product, HNSW/IVF concepts.
- Visual: nearest-neighbor search.

### S139 - Vector Stores [Core]

- Core message: A vector database stores vectors plus metadata and search indexes.
- Slide content: Chroma, FAISS, Qdrant, Weaviate, Pinecone, pgvector.
- Visual: local to managed spectrum.

### S140 - Metadata Is a Feature [Core]

- Core message: Metadata filters often matter as much as vector similarity.
- Slide content: source, date, permission, product, region, document type.
- Visual: retrieval with filter.

### S141 - Access Control in RAG [Core]

- Core message: Retrieval must respect user permissions before generation.
- Slide content: ACL filtering, tenant isolation, source-level authorization, audit logs.
- Visual: user-specific retrieval gate.

### S142 - Hybrid Search [Core]

- Core message: Semantic search and keyword search catch different things.
- Slide content: dense vectors plus BM25; combine and rerank.
- Visual: two retrieval streams merging.

### S143 - Reranking [Core]

- Core message: Retrieve broad, then rerank narrow.
- Slide content: top-20 retrieval, cross-encoder/reranker, final top-3 context.
- Visual: candidate funnel.

### S144 - Query Rewriting [Backup]

- Core message: Rewriting improves retrieval when user questions are vague or conversational.
- Slide content: standalone question, HyDE, expansion, decomposition.
- Visual: query transformation.

### S145 - Parent-Child and Hierarchical RAG [Core]

- Core message: Retrieve small, answer with larger context.
- Slide content: child chunks for matching; parent sections for generation.
- Visual: child-to-parent link.

### S146 - Multi-Step and Agentic RAG [Core]

- Core message: Some questions require planning, multiple retrievals, and verification.
- Slide content: decompose, retrieve, verify, answer, cite.
- Visual: loop.
- Talk track: Do not use agentic RAG for simple FAQ.

### S147 - Graph RAG Awareness [Backup]

- Core message: Graphs help when relationships matter.
- Slide content: entities, relationships, communities, structured retrieval.
- Visual: knowledge graph plus documents.

### S148 - RAG Prompting [Core]

- Core message: The prompt must force grounded behavior.
- Slide content: answer only from context, cite sources, say insufficient context, separate facts from reasoning.
- Visual: prompt template.

### S149 - RAG Evaluation Dimensions [Core]

- Core message: RAG quality is multi-dimensional.
- Slide content: context precision, context recall, faithfulness, answer relevance, correctness, latency, cost.
- Citation: RAGAS paper.

### S150 - Golden Datasets [Core]

- Core message: You need stable questions to compare pipeline changes.
- Slide content: question, ideal answer, source doc, expected citations, difficulty label.
- Visual: eval dataset row.

### S151 - LLM-as-a-Judge [Core]

- Core message: Judge models can scale evaluation but need rubrics and calibration.
- Slide content: rubric, pairwise compare, score with rationale, human spot checks.
- Visual: judge pipeline.

### S152 - RAGAS and Automated Evals [Core]

- Core message: Frameworks speed up reference-free and reference-based RAG evaluation.
- Slide content: faithfulness, answer relevancy, context precision/recall.
- Citation: RAGAS paper.
- Lab hook: Lab 5 optional RAGAS evaluation.

### S153 - RAG Observability [Core]

- Core message: Log every stage or you cannot debug failures.
- Slide content: query, rewritten query, retrieved chunk IDs, scores, prompt tokens, answer, judge score, latency, cost.
- Visual: trace waterfall.

### S154 - Prompt Injection in RAG [Core]

- Core message: Retrieved text can contain malicious instructions.
- Slide content: untrusted documents, instruction hierarchy, content sanitization, allowlists, tool permissioning.
- Visual: malicious document entering context.

### S155 - RAG Security Checklist [Handout]

- Core message: RAG needs data security, not only model safety.
- Slide content: ACLs, source logging, PII handling, prompt injection tests, citation policy, deletion propagation.
- Visual: checklist.

### S156 - RAG Anti-Patterns [Core]

- Core message: Most RAG failures are predictable.
- Slide content: giant chunks, no evals, no metadata, no citations, one embedding model forever, no freshness strategy, reranking everything.
- Visual: anti-pattern list.

### S157 - RAG Maturity Levels [Core]

- Core message: RAG evolves from demo to monitored knowledge system.
- Slide content: naive vector search, metadata filters, hybrid/rerank, eval-driven, permissioned/observed, agentic/graph when needed.
- Visual: maturity ladder.

### S158 - Activity: Diagnose the RAG Failure [Activity]

- Core message: Students learn stage-specific debugging.
- Slide content: wrong source, missing source, hallucinated answer, slow answer, no citation, user sees forbidden doc.
- Visual: diagnosis table.

### S159 - Part 7 Takeaways [Core]

- Core message: Production RAG is a measurement and data pipeline problem.
- Slide content: chunk carefully, retrieve with metadata, rerank when useful, evaluate continuously, secure retrieval before generation.

---

# Part 8: Deploying and Showcasing

Goal: make the system visible, shareable, routed, and measurable.

### S160 - From API to User Experience [Core]

- Core message: Users do not experience a model; they experience a product surface.
- Slide content: chat UI, sources, latency, error states, feedback, trace IDs.
- Visual: product surface over service.

### S161 - Gradio for ML Demos [Core]

- Core message: Gradio is the fastest path to feedback.
- Slide content: `ChatInterface`, `Blocks`, examples, sliders, share links.
- Visual: Gradio screen.
- Lab hook: Lab 6.

### S162 - Streaming in the UI [Core]

- Core message: Streaming turns waiting into progress.
- Slide content: token streaming, cancel, partial answer, loading states.
- Visual: streaming chat.

### S163 - Showing Sources [Core]

- Core message: RAG UIs should expose the evidence trail.
- Slide content: retrieved snippets, source labels, citations, confidence caveat.
- Visual: answer plus source panel.

### S164 - Hugging Face Spaces Workflow [Core]

- Core message: Spaces turns a demo into a persistent shareable artifact.
- Slide content: create Space, SDK Gradio, `app.py`, `requirements.txt`, secrets, hardware.
- Visual: git push to URL.

### S165 - ZeroGPU Awareness [Backup]

- Core message: ZeroGPU reduces barriers for GPU-backed demos but has eligibility and quota rules.
- Slide content: free use of existing ZeroGPU Spaces; hosting requires PRO/Team/Enterprise per HF docs.
- Citation: HF ZeroGPU docs.

### S166 - LiteLLM Gateway Pattern [Core]

- Core message: A gateway centralizes routing, cost, keys, fallbacks, and observability.
- Slide content: app -> LiteLLM -> OpenAI/Groq/vLLM/HF/Azure.
- Visual: gateway architecture.
- Citation: LiteLLM docs.

### S167 - Routing Policies [Core]

- Core message: Route by cost, latency, capability, privacy, or fallback.
- Slide content: simple -> cheap model; complex -> stronger model; failure -> backup; sensitive -> private endpoint.
- Visual: router rules.

### S168 - Cost Tracking [Core]

- Core message: Cost observability is part of production readiness.
- Slide content: cost/request, cost/user, cost/team, budget alerts, model mix.
- Visual: cost dashboard.

### S169 - Observability: What to Trace [Core]

- Core message: Trace the full request path, not just the final answer.
- Slide content: user request, retrieved chunks, model call, tool calls, latency, tokens, cost, output, eval score.
- Visual: trace waterfall.
- Citation: Langfuse docs.

### S170 - Langfuse, MLflow, Phoenix, LangSmith [Backup]

- Core message: Observability tools differ, but the data model is similar.
- Slide content: traces, spans, datasets, scores, prompt versions, dashboards.
- Visual: tool landscape.

### S171 - Feedback Loops [Core]

- Core message: Production systems improve through measurement loops.
- Slide content: user feedback, judge scores, offline evals, prompt/model changes, regression tests.
- Visual: eval flywheel.

### S172 - Prompt and Model Versioning [Core]

- Core message: Every answer should be attributable to versions.
- Slide content: prompt ID, model ID, embedding model, retriever config, chunker version, deployment version.
- Visual: request metadata card.

### S173 - Deployment Readiness Checklist [Handout]

- Core message: A live URL is not production.
- Slide content: auth, secrets, rate limits, logs, evals, fallback, source display, privacy, cost cap, rollback.
- Visual: checklist.

### S174 - Demo, MVP, or Production? [Activity]

- Core message: Students classify deployments based on controls.
- Slide content: Gradio share link, internal Space, FastAPI endpoint, vLLM behind gateway, enterprise RAG assistant.
- Visual: maturity cards.

### S175 - Part 8 Takeaways [Core]

- Core message: Shipping means interface, routing, observability, and iteration.
- Slide content: UI for feedback, Spaces for showcasing, gateways for control, traces for debugging.

---

# Part 9: Putting It All Together

Goal: synthesize into decisions students can reuse after the class.

### S176 - We Started with a Model; We End with a System [Core]

- Core message: The class journey is a stack, not isolated techniques.
- Slide content: model -> API -> RAG -> UI -> gateway -> observability -> governance.
- Visual: completed architecture.

### S177 - Complete Reference Architecture [Core]

- Core message: A production-pattern LLM app has explicit layers.
- Slide content: UI, API, gateway, orchestration, retrieval, serving/provider, data, evals, observability, governance.
- Visual: full reference architecture.

### S178 - Pattern 1: API-Based LLM App [Core]

- Core message: Use hosted APIs when speed, quality, and low ops matter.
- Slide content: app -> OpenAI/Groq/provider; add logging, evals, fallbacks.
- Best for: prototypes, low ops teams, frontier capability.

### S179 - Pattern 2: Private Open-Model Serving [Core]

- Core message: Self-host when privacy, cost at scale, or control dominate.
- Slide content: app -> gateway -> vLLM/TGI/SGLang -> GPU.
- Best for: sensitive data, predictable volume, open-model strategy.

### S180 - Pattern 3: Enterprise RAG Assistant [Core]

- Core message: Enterprise RAG is a permissioned retrieval and evaluation system.
- Slide content: connectors, ACLs, chunking, vector/keyword search, rerank, grounded generation, audit logs.
- Best for: internal knowledge, support, policy, research.

### S181 - Pattern 4: Gateway-Based Platform [Core]

- Core message: Gateways let platform teams support many applications.
- Slide content: central keys, budgets, routing, observability, provider abstraction.
- Best for: multi-team organizations.

### S182 - Pattern 5: Adapted Model Deployment [Backup]

- Core message: Fine-tuned adapters fit stable repeated behavior.
- Slide content: base model, adapter registry, eval gate, serve adapter, monitor drift.
- Best for: domain format, tone, specialized extraction, repeated workflows.

### S183 - Decision Playbook: Primary Constraint [Core]

- Core message: Start decisions from the constraint.
- Slide content: cost, latency, quality, privacy, freshness, governance, developer speed.
- Visual: constraint compass.

### S184 - Prompt, RAG, or Fine-Tune? [Handout]

- Core message: Choose adaptation by failure mode.
- Slide content: instruction issue -> prompt; missing/changing facts -> RAG; stable repeated behavior -> fine-tune; all three -> hybrid.
- Visual: decision tree.

### S185 - Quantize, Distill, Prune, or Choose Smaller? [Handout]

- Core message: Efficiency choices should be reversible and measured.
- Slide content: choose smaller, quantize, optimize serving, cache/route, distill selectively, prune rarely.
- Visual: decision table.

### S186 - FastAPI, vLLM, TGI, Ollama, Managed API? [Handout]

- Core message: Serving engine follows workload.
- Slide content: FastAPI for control; vLLM for throughput; TGI for HF enterprise; Ollama for local dev; managed API for low ops/frontier.
- Visual: selection matrix.

### S187 - RAG Design Playbook [Handout]

- Core message: RAG decisions are data decisions.
- Slide content: source, permissions, chunking, embeddings, vector store, hybrid, rerank, prompt, eval, monitoring.
- Visual: RAG design checklist.

### S188 - Architecture Decision Records [Core]

- Core message: Deployment decisions should be written down.
- Slide content: context, decision, alternatives, consequences, metrics, review date.
- Visual: ADR template.

### S189 - Enterprise Readiness [Core]

- Core message: Enterprise readiness is not one checkbox.
- Slide content: security, privacy, compliance, reliability, evaluation, auditability, cost, human escalation.
- Visual: readiness matrix.

### S190 - Guardrails: What They Are and Are Not [Core]

- Core message: Guardrails reduce risk but do not replace design and evaluation.
- Slide content: input filters, output validation, policy checks, tool permissions, human review.
- Visual: layered controls.

### S191 - NIST AI RMF and GenAI Profile [Core]

- Core message: Governance needs a shared risk language.
- Slide content: govern, map, measure, manage; GenAI risks such as hallucination, misuse, data leakage, bias, security.
- Citation: NIST AI RMF GenAI Profile.

### S192 - EU AI Act Awareness [Backup]

- Core message: Regulation depends on use case risk, not just model choice.
- Slide content: prohibited, high-risk, transparency, GPAI obligations awareness.
- Visual: risk tier.
- Talk track: This is awareness, not legal advice.

### S193 - Red Teaming and Adversarial Testing [Core]

- Core message: Test hostile and weird inputs before users do.
- Slide content: jailbreaks, prompt injection, PII exfiltration, tool misuse, malformed data, multilingual attacks.
- Visual: attack test suite.

### S194 - Human-in-the-Loop Patterns [Backup]

- Core message: High-stakes systems need escalation and review.
- Slide content: approval queue, confidence thresholds, reviewer feedback, audit trails.
- Visual: escalation flow.

### S195 - Production Readiness Scorecard [Handout]

- Core message: Score the system before calling it production.
- Slide content: quality, latency, cost, security, observability, reliability, governance, ownership.
- Visual: radar chart.

### S196 - Learning Paths by Role [Core]

- Core message: Students leave with different next steps.
- Slide content: engineer, data scientist, architect/platform, product/leader.
- Visual: role map.

### S197 - Recommended Practice Projects [Core]

- Core message: Skill sticks when applied to real mini-systems.
- Slide content: internal policy RAG, model router, adapter demo, eval harness, cost dashboard, prompt injection test suite.
- Visual: project cards.

### S198 - Final Mental Model [Core]

- Core message: The work is to match constraints to architecture.
- Slide content: "Use the smallest reliable model, grounded in the right context, served through a measurable interface, with a path to change safely."
- Visual: one-sentence poster.

### S199 - Capstone Brief [Core]

- Core message: Students now prove the stack through a domain system.
- Slide content: build a domain RAG assistant or fine-tuned showcase; present problem, architecture, demo, tradeoffs.
- Visual: capstone checklist.

### S200 - Closing [Core]

- Core message: Mastery is not memorizing tools; it is knowing which lever to pull and how to measure the result.
- Slide content: "Models will change. The deployment judgment transfers."
- Visual: completed course journey.
