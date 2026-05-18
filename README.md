# Mastering LLM Deployment

A 2-day hands-on course for software engineers and data scientists on deploying Large Language Models efficiently and at scale.

![Gemini_Generated_Image_6winox6winox6win](images/Gemini_Generated_Image_6winox6winox6win.png)

## Course Objectives

- Understand the modern GenAI stack (HuggingFace, LangChain, OpenAI-compatible APIs)
- Inspect transformer model internals and manage multi-turn conversations
- Quantize models with bitsandbytes (NF4/INT4) and fine-tune with QLoRA
- Build and serve OpenAI-compatible APIs with FastAPI, understand vLLM at scale
- Build a full RAG pipeline with local embeddings, ChromaDB, and LLM-as-Judge evaluation
- Ship a streaming Gradio app backed by RAG

## Getting Started

### Google Colab (Recommended)

Click any **Open in Colab** badge below to launch a notebook directly. No local setup needed.

**You will need an OpenAI API key** (provided by the instructor) for Labs 1, 4, 5, and 6.

> **Enabling GPU for Lab 3 and Capstone Track B:**
> In Colab, go to **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**, then re-run from the top.

### Local Setup

```bash
pip install -r requirements.txt
jupyter lab
```

## Lab Structure

| Lab | Topic | Duration | Hardware | Open in Colab |
|-----|-------|----------|----------|---------------|
| **Lab 0** — Environment Check | Verify all dependencies install and run | 10 min | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/00_Environment_Check/lab0_env_check.ipynb) |
| **Lab 1** — Modern Stack | HuggingFace internals, OpenAI SDK, `base_url` swap, LangChain chains | 45 min | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/01_Modern_Stack/lab1_modern_stack.ipynb) |
| **Lab 2** — Inspect & Chat | Model architecture, tokenization, generation controls, multi-turn sessions | 45 min | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/02_Inspect_Chat/lab2_inspect_chat.ipynb) |
| **Lab 3** — Quantize + LoRA | INT4/NF4 quantization, LoRA mechanics, QLoRA fine-tuning | 75 min | **T4 GPU** ⚡ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/03_Quantize_LoRA/lab3_quantize_lora.ipynb) |
| **Lab 4** — Serving API | FastAPI OpenAI-compatible server, ngrok tunnels, vLLM deep dive | 45 min | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/04_Serving_API/lab4_serving_api.ipynb) |
| **Lab 5** — RAG Pipeline | Chunk → embed → retrieve → generate → RAGAS evaluation | 60 min | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/05_RAG_Pipeline/lab5_rag_pipeline.ipynb) |
| **Lab 6** — Gradio RAG App | Streaming Gradio app, partner red-team challenge | 45 min | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/06_Gradio_RAG_App/lab6_gradio_rag_app.ipynb) |
| **Capstone** | Domain RAG Assistant (CPU) or Fine-Tuned Showcase (T4 GPU) | 3 hr | CPU / T4 ⚡ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Capstone/capstone_project.ipynb) |

> ⚡ **Lab 3 and Capstone Track B require T4 GPU.** All other labs run on the free CPU runtime.

## Day Schedule

**Day 1**
- AM: Lab 0 (pre-class) → Lab 1 → Lab 2
- PM: Lab 3

**Day 2**
- AM: Lab 4 → Lab 5
- PM: Lab 6 → Capstone

## Key Dependencies

```
transformers torch bitsandbytes peft trl
openai langchain langchain-openai
sentence-transformers chromadb
fastapi uvicorn pyngrok
gradio
ragas
```

See `requirements.txt` for pinned versions.

## Prerequisites

- Python proficiency
- Basic ML/deep learning concepts
- No prior LLM deployment experience required
