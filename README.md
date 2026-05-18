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

Open any notebook directly in Colab by changing `github.com` → `githubtocolab.com` in the URL, or use the badge links below.

**You will need an OpenAI API key** (provided by the instructor) for Labs 1, 4, 5, and 6.

### Local Setup

```bash
pip install -r requirements.txt
jupyter lab
```

## Lab Structure

| Lab | Folder | Topic | Duration | Hardware |
|-----|--------|-------|----------|----------|
| 0 | `00_Environment_Check` | Environment sanity check — verify all dependencies install and run | 10 min | CPU |
| 1 | `01_Modern_Stack` | The GenAI stack: HuggingFace internals, OpenAI SDK, base_url swap, LangChain chains | 45 min | CPU |
| 2 | `02_Inspect_Chat` | Model architecture, tokenization, generation controls, multi-turn chat sessions | 45 min | CPU |
| 3 | `03_Quantize_LoRA` | INT4/NF4 quantization with bitsandbytes, LoRA mechanics, QLoRA fine-tuning | 75 min | **T4 GPU** |
| 4 | `04_Serving_API` | FastAPI OpenAI-compatible server, ngrok tunnels, vLLM conceptual deep dive | 45 min | CPU |
| 5 | `05_RAG_Pipeline` | Full RAG pipeline: chunk → embed → retrieve → generate → evaluate with RAGAS | 60 min | CPU |
| 6 | `06_Gradio_RAG_App` | Streaming Gradio RAG app, partner red-team challenge | 45 min | CPU |
| — | `Capstone` | Build your own: Domain RAG Assistant (CPU) or Fine-Tuned Model Showcase (GPU) | 3 hr | varies |

> **Lab 3 requires a T4 GPU runtime.** All other labs run on CPU (Colab free tier).

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
