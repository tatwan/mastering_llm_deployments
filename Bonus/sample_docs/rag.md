# Retrieval-Augmented Generation

RAG retrieves relevant source chunks at inference time, injects them into the prompt, and asks the model to answer only from that context.

Use RAG when knowledge changes or lives outside the weights (policies, docs, tickets). Use fine-tuning when you need style, format, or a skill baked into the model.

Lab 6 builds RAG by hand: chunk, MiniLM embeddings, Chroma, grounded prompt. This bonus uses LlamaIndex to do the same pipeline in fewer lines.
