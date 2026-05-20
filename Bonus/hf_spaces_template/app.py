import os
import time

import chromadb
import gradio as gr
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it as a Hugging Face Space secret.")

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = "all-MiniLM-L6-v2"

oai = OpenAI(api_key=OPENAI_API_KEY)

knowledge_base = {
    "quantization": (
        "Quantization reduces model weight precision. NF4 is a 4-bit format designed for "
        "normally distributed LLM weights. It lowers memory use versus FP16 while preserving "
        "enough quality for many deployment tasks."
    ),
    "rag": (
        "Retrieval-Augmented Generation retrieves relevant source chunks at inference time, "
        "injects them into the prompt, and asks the model to answer only from that context. "
        "RAG is best for changing or external knowledge."
    ),
    "lora": (
        "LoRA freezes the base model and trains small low-rank adapter matrices. QLoRA combines "
        "a 4-bit quantized base model with trainable LoRA adapters, making fine-tuning possible "
        "on smaller GPUs."
    ),
    "serving": (
        "OpenAI-compatible endpoints let applications swap backends by changing base_url. "
        "vLLM improves throughput with PagedAttention and continuous batching."
    ),
}


def build_collection():
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.Client()
    collection = client.get_or_create_collection("course_kb", embedding_function=embedding_fn)
    existing = collection.get().get("ids", [])
    if existing:
        collection.delete(ids=existing)

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
    docs = [
        Document(page_content=text, metadata={"source": source})
        for source, text in knowledge_base.items()
    ]
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk-{i}"],
            documents=[chunk.page_content],
            metadatas=[{"source": chunk.metadata["source"]}],
        )
    return collection


collection = build_collection()

SYSTEM_PROMPT = """You are a course assistant for Mastering LLM Deployment.
Answer only from the provided context.
If the context does not contain the answer, say: "The knowledge base does not cover that."
Keep answers concise and cite the source topic names."""


def retrieve(question, n=3):
    result = collection.query(query_texts=[question], n_results=n)
    return result["documents"][0], result["metadatas"][0]


def answer(question):
    t0 = time.time()
    docs, metas = retrieve(question)
    context = "\n\n".join(
        f"[{meta['source']}]: {doc}" for doc, meta in zip(docs, metas)
    )
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = oai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    text = response.choices[0].message.content
    latency = int((time.time() - t0) * 1000)
    sources = ", ".join(sorted({meta["source"] for meta in metas}))
    return f"{text}\n\nSources: {sources}\nLatency: {latency} ms"


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Question", placeholder="What is QLoRA?"),
    outputs=gr.Markdown(label="Answer"),
    title="Master LLM Deployment RAG Demo",
    examples=[
        "What is QLoRA?",
        "Why does vLLM help throughput?",
        "When should I use RAG instead of fine-tuning?",
        "What is the weather today?",
    ],
)


if __name__ == "__main__":
    demo.launch()

