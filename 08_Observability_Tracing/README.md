# Lab 8 - Observability and Tracing for RAG


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/08_Observability_Tracing/lab8_observability.ipynb)
**Production Readiness Pack | ~60-75 minutes | CPU | OpenAI API key optional**

---

## Purpose

MLflow is one common way to track RAG experiments and evaluation runs. Whether or not you use MLflow, the production idea is the same: a RAG system is not finished when it answers one question. You need a repeatable way to track experiments, evaluate quality, and explain failures.

This lab continues that thread. MLflow is excellent for experiment tracking and evaluation runs. Here we zoom in on **request-level observability**: when one user asks one question, what happened inside the system?

By the end, students can answer:

- Did retrieval return the right chunks?
- Did the prompt contain enough grounding context?
- Was latency caused by embedding, retrieval, or generation?
- Did an out-of-scope or prompt-injection question fail because of retrieval, prompting, or generation?

## What You Will Build

1. A tiny RAG pipeline similar to Labs 6 and 7.
2. A manual trace recorder that works reliably in Colab.
3. A trace dataframe showing spans, latency, retrieved chunks, prompt previews, and output.
4. Three diagnostic runs: in-scope, out-of-scope, and prompt-injection attempt.
5. An optional Phoenix/OpenTelemetry extension for students who want a real observability UI.

## Why This Lab Exists

Lab 6 taught RAG quality. Lab 7 taught product behavior. MLflow-style tracking and LLM-as-a-Judge evaluation answer run-level quality questions. Lab 8 teaches the missing operational question:

> A user received a bad answer. How do I reconstruct what happened?

## MLflow vs Tracing

| Need | Better Tooling Pattern |
| --- | --- |
| Compare chunk sizes, prompts, retrieval k, and model choices | MLflow experiment tracking |
| Score a golden dataset across many questions | MLflow, RAGAS, or another eval harness |
| Debug one bad production answer | Request tracing with spans |
| See retrieved documents and prompt assembly for one request | Phoenix, LangSmith, MLflow traces, or custom trace logs |

## Key Terms

| Term | Definition |
| --- | --- |
| Trace | The full journey of one request through the system |
| Span | One timed operation inside a trace, such as embedding or retrieval |
| Attribute | Metadata attached to a span, such as model name or retrieved chunk IDs |
| Experiment tracking | Comparing many runs/configurations over time |
| Golden dataset | A stable set of test questions used to evaluate regressions |
| Failure diagnosis | Explaining whether a bad answer came from retrieval, prompt assembly, or generation |

## Where MLflow Fits

MLflow fits as a bridge and optional artifact logger. If you have seen an MLflow UI, this lab gives that experience a clear mental model: MLflow-style evaluation compares configurations across a dataset, while request tracing explains one live request. The notebook includes an optional MLflow cell that logs trace tables and diagnosis artifacts. Phoenix remains optional because it gives a clearer span-inspection UI, but the core concept is tool-neutral.

## Reusing Lab 6

Lab 6 creates a persistent ChromaDB at `./chroma_db` with collection `llm_course`. Lab 8 includes an optional adapter to load that database when it exists. The tiny built-in corpus stays in the main path so the lab remains self-contained in a fresh Colab runtime.

## Instructor Notes

If time is short, teach this as a guided demo after Lab 7. If students explore later, the notebook is designed to run even without Phoenix. The manual trace recorder is intentionally simple so the concept survives tooling/version changes.
