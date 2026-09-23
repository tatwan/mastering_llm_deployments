# Lab 9 — Semantic Caching and Cost Control

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/09_Semantic_Caching/lab9_semantic_caching.ipynb)

**Production Readiness Pack | ~45 minutes | CPU | `OPENAI_API_KEY` recommended (a scripted stand-in runs without one)**

---

## Coming from Lab 8

A trace tells you what one request did and what it cost. This lab tries not to pay for the next one.

---

## Why this lab exists

People ask the same question in different words all day. A semantic cache embeds each question and reuses a stored answer when a new one means nearly the same thing. It cuts cost and latency, and it adds a new failure: a fast, confident answer to a question nobody asked.

The notebook answers from a five-line FAQ, so a wrong answer is easy to spot. Two of those lines are the trap: premium customers get a 30-day refund, enterprise customers do not.

## What you will do

1. Build an exact cache and watch a paraphrase miss it.
2. Build a semantic cache whose entries record model, FAQ version, time and scope.
3. Watch the enterprise customer get the premium refund policy, in milliseconds.
4. Look at six real similarity scores and see that **no threshold** separates "sounds alike" from "has the same answer". A CPU question scores higher than a genuine paraphrase.
5. Fix it the way production systems do: put the customer's plan in the cache key.
6. Add up what the cache saved, from the token counts the API returns.

## The idea to keep

Similarity is not equivalence. Embeddings measure how alike two questions *sound*; whether they deserve the same answer is a business rule. The facts that decide it (plan, tenant, region, permissions) usually are not in the question at all. They come from the session, and they belong in the cache key.

## Key terms

| Term | Meaning |
| --- | --- |
| Exact cache | Keyed by the exact question string |
| Semantic cache | Keyed by embedding similarity between questions |
| False miss | A reusable answer is not found (the paraphrase at 0.76) |
| False hit | A stored answer is reused for a question that needed a different one |
| Scope | Who an answer is valid for: plan, tenant, user, region |
| TTL | How long an entry stays valid |
| Document version | Which version of the source material produced the answer |

## Next

[Lab 10 — Containerization](../10_Containerization/README.md). It needs Docker on your own machine; Colab cannot finish the build.
