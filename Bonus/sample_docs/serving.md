# Serving

The OpenAI Chat Completions format is the wire protocol of this course. Change `base_url` to point the same client at OpenAI, Groq, a student FastAPI proxy, vLLM, or Ollama.

vLLM improves throughput with PagedAttention and continuous batching. A naive FastAPI proxy (Lab 5) is the right classroom server; vLLM is what you run on a GPU box at scale.

Gradio `share=True` (Lab 7) is a temporary URL. Hugging Face Spaces (Bonus 07) is the persistent one.
