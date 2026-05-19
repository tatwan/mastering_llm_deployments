# Bonus Notebooks

These notebooks go deeper on topics introduced in the main labs. They are optional but recommended for students who want to explore further.

---

## Notebooks

### 01 — Function Calling (`02_function_calling.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/02_function_calling.ipynb)

Hands-on practice with the OpenAI function calling API. You'll build a weather lookup agent and a DuckDB-backed data analysis agent. Covers the full 2-round loop: model decides which tool to call → you execute it → model receives the result and answers.

**Prerequisite:** Lab 1 (Modern Stack)  
**Runtime:** CPU | OpenAI API key required

---

### 02 — RAG with LlamaIndex (`02_rag_llamaindex.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/02_rag_llamaindex.ipynb)

Build the same PDF RAG pipeline you saw in Lab 6, but using LlamaIndex instead of LangChain + ChromaDB. Compare how LlamaIndex abstracts away chunking, embedding, and retrieval with just a few lines of code. Covers index persistence to disk and loading from disk.

**Prerequisite:** Lab 6 (RAG Pipeline)  
**Runtime:** CPU | OpenAI API key required

---

### 03 — ReAct Agent from Scratch (`04_react_agent.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/Bonus/04_react_agent.ipynb)

Implement the Thought → Action → Observation loop by hand using raw OpenAI API calls — no frameworks. You'll see exactly what happens inside every LangChain agent. Ends with a discussion of the Model Context Protocol (MCP), the emerging standard that replaces custom tool wiring.

**Prerequisite:** Lab 1 (Modern Stack), Lab 2 (Prompting)  
**Runtime:** CPU | OpenAI API key required

---

## When to Use These

| If you want to… | Open this notebook |
|---|---|
| Understand how function calling works under the hood | `02_function_calling.ipynb` |
| See a simpler alternative to LangChain for RAG | `02_rag_llamaindex.ipynb` |
| Understand how AI agents actually work | `04_react_agent.ipynb` |
