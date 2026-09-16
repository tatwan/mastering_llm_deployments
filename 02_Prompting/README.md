# Lab 2 — Prompting Fundamentals & Responsible AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/02_Prompting/lab2_prompting.ipynb)

**Day 1 Morning | ~45 minutes | CPU | OpenAI API key required**

---

## Coming from Lab 1

Lab 1A gave you `OpenAI(api_key=..., base_url=...)`. Lab 1B showed tool calls as extra message roles. This lab is the **text you send**: prompts as software, including how they fail under injection.

Same Colab Secret: `OPENAI_API_KEY`. Same workhorse: `gpt-4o-mini`.

---

## Purpose

Prompting is the primary interface between your application logic and the model. A poorly designed prompt produces inconsistent, unparseable, or dangerous outputs. A well-designed one is deterministic, testable, and secure. This lab teaches the patterns every LLM deployment engineer needs — including the security patterns that most tutorials skip.

---

## What You Will Build

**Part A — Anatomy of a Prompt**
You'll compare a vague "bad" prompt against a structured "good" prompt on a real LLM deployment question. The four-component anatomy (Role, Context, Task, Format) is a checklist, not a rule — you'll see which components matter most for which situations.

**Part B — Prompting Patterns**
You'll implement and compare three patterns on deployment-relevant tasks:
- **Zero-shot** — classify a deployment question with no examples
- **Few-shot** — teach a custom classification scheme with 4 examples; watch how the model generalizes
- **Chain-of-thought** — ask the same deployment decision question with and without step-by-step reasoning; compare the quality

You'll also build a structured JSON extractor (model returns parseable config objects) and a persona demo (same question, three system prompts → three different recommendation philosophies).

**Part C — Responsible Prompting**
The section most tutorials skip. You will attack a support bot whose user text is concatenated into the prompt, then harden it with the `system` role. Then **indirect** injection: a poisoned RAG chunk (preview of Labs 6–7). Then prompt leaking — extracting a system prompt that should never have held a secret.

---

## Critical Points

**Prompting is software engineering, not magic.** The same debugging mindset applies: isolate variables, test edge cases, version your prompts.

**Zero-shot first.** Don't add examples or CoT by default. Each technique adds tokens (cost and latency). Add complexity only when you've confirmed the simpler approach fails.

**The system role is a trust boundary, not a suggestion.** When you put instructions in the system role and user input in the user role, the model treats them differently. When you concatenate user input into a prompt string, it doesn't. This difference is the entire basis of prompt injection defense.

**Perfect injection defense doesn't exist.** Even well-designed system prompts can be bypassed with sophisticated attacks (jailbreaking, encoding tricks, role-play framing). The goal is to make attacks expensive and detectable, not to make them impossible. Defense in depth: system role separation + output validation + monitoring.

**Structured output fails silently.** When you ask for JSON, the model might add markdown fences, add an explanation, or occasionally return malformed JSON. Always include a parse attempt with a fallback. In production, add retry logic.

---

## Connection to the Rest of the Course

- **Lab 5 (Serving API):** Your FastAPI server's system prompt is a deployment artifact — version it, test it, secure it exactly like you would code
- **Lab 6 (RAG Pipeline):** The grounding instruction ("Answer ONLY based on the provided context") is both a prompting pattern and a security control against hallucination
- **Lab 7 (Gradio App):** The partner red-team challenge is a structured prompt injection exercise — you'll attack each other's RAG app system prompts
- **Lab 3 (Inspect & Chat):** Multi-turn history is explicit — you must resend messages. Lab 3 builds that into a `ChatSession` on a local model.

---

## Key Terms

| Term | Definition |
|------|-----------|
| Zero-shot | Asking the model to perform a task without examples |
| Few-shot | Providing example input→output pairs to teach a pattern |
| Chain-of-thought | Instructing the model to reason step-by-step before answering |
| System role | The `system` message in the OpenAI chat format — the highest-trust channel for instructions |
| Prompt injection | A crafted input that overrides or hijacks the model's original instructions |
| Prompt leaking | A crafted input that extracts the system prompt from the model's response |
| Structured output | Prompting the model to return a specific parseable format (JSON, XML, CSV) |

---

## Student exercises (in the notebook)

Two TODO cells at the end are **intentional**:

1. Write a CoT prompt that picks FastAPI / vLLM / Ollama for a 13B / 2×A100 scenario.
2. Rewrite `vulnerable_helpdesk` using `system=` (the function starts as `NotImplementedError`).

---

## Next

[Lab 3 — Inspect & Chat](../03_Inspect_Chat/README.md) — local Qwen2.5-0.5B, no API key, architecture numbers and a real chat loop.
