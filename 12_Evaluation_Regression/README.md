# Lab 12 - Evaluation and Regression Testing


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/12_Evaluation_Regression/lab12_evaluation_regression.ipynb)
**Production Readiness Pack | ~60 minutes | CPU | OpenAI API key optional**

---

## Purpose

A deployed LLM system changes constantly: prompts, models, documents, chunking, retrieval `k`, guardrails, and dependencies. Without regression tests, a small improvement can silently break another behavior.

This lab turns the evaluation-tracking idea into a lightweight deployment regression harness students can reuse in the Capstone.

## What You Will Build

1. A golden dataset with in-scope, out-of-scope, and attack questions.
2. Deterministic checks for must-include and must-not-include behavior.
3. A pass/fail report comparing two RAG configurations.
4. Optional MLflow logging for run comparison.
5. A Capstone-ready evaluation checklist.

## Why Not Only LLM-as-a-Judge?

LLM judges are useful, but they cost money and can change behavior. Production teams often start with cheap deterministic checks:

- Did the answer cite a source?
- Did it decline out-of-scope questions?
- Did it avoid forbidden strings?
- Did it include required terms?

Then they add LLM-as-a-Judge for nuanced quality evaluation.

## Where MLflow Fits

MLflow-style evaluation tracking is still valuable, but students do not need a separate MLflow lab before this one. This lab uses a robust local harness first, then includes optional MLflow logging so students can compare runs and preserve artifacts. The teaching sequence is: deterministic checks first, MLflow/RAGAS/LLM-as-a-Judge when semantic grading is worth the extra cost and tool complexity.

## Key Terms

| Term | Definition |
| --- | --- |
| Golden dataset | Stable set of representative test questions |
| Regression | A change that breaks behavior that used to work |
| Deterministic check | Rule-based pass/fail test that does not call a judge model |
| Prompt version | Named version of the prompt used for a run |
| Eval harness | Code that runs test cases and reports pass/fail results |
