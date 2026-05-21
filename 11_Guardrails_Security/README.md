# Lab 11 - Guardrails and Deployment Security


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/11_Guardrails_Security/lab11_guardrails_security.ipynb)
**Production Readiness Pack | ~60 minutes | CPU | No API key required**

---

## Purpose

Lab 7 asks students to red-team each other's RAG apps. This lab answers the next question:

> We broke it. How do we make it safer before users touch it?

Students build lightweight, programmatic guardrails around a mini RAG system. The point is not to promise perfect security. The point is to teach layered defenses and honest residual risk.

## What You Will Build

1. A vulnerable mini RAG responder.
2. An input guard for prompt-injection patterns.
3. A retrieval confidence gate for out-of-scope questions.
4. An output guard that redacts PII-like strings and fake secrets.
5. An attack suite that compares before/after behavior.

## Key Terms

| Term | Definition |
| --- | --- |
| Prompt injection | User input that attempts to override system/developer instructions |
| Guardrail | Programmatic check before or after the LLM call |
| Retrieval gate | Minimum relevance threshold before generation is allowed |
| Output redaction | Removing sensitive data from generated text before returning it |
| Defense in depth | Multiple imperfect controls layered together |
| Residual risk | The risk that remains after controls are added |

## Frameworks and Tools

This lab starts with lightweight guards so students understand the control points: input checks, retrieval confidence, output redaction, logging, and residual risk. The notebook then maps those control points to production tools such as Guardrails AI, NeMo Guardrails, Llama Guard, Microsoft Presidio, and provider content filters.

## Instructor Notes

Keep this practical. Students should leave knowing that system prompts help, but production safety also needs input checks, retrieval confidence, output validation, logging, and human review for high-risk domains.
