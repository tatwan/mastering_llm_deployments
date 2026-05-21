# Lab 9 - Semantic Caching and Cost Control


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/09_Semantic_Caching/lab9_semantic_caching.ipynb)
**Production Readiness Pack | ~45-60 minutes | CPU | OpenAI API key optional**

---

## Purpose

Serving an LLM app means paying for repeated work. If many users ask equivalent public FAQ questions, caching can reduce latency and cost. But semantic caching is not automatically safe: similar questions can require different answers.

This lab teaches caching as a deployment tradeoff, not just a speed trick.

## What You Will Build

1. An exact cache that only matches identical strings.
2. A semantic cache using local embeddings and cosine similarity.
3. A benchmark table comparing no cache, exact cache, and semantic cache.
4. A threshold-tuning exercise that reveals false misses and false hits.
5. Cache metadata checks for model version, document version, and TTL.

## Student Questions This Lab Answers

- Why does normal Redis-style exact caching miss paraphrases?
- How can embeddings make cache lookup semantic?
- What can go wrong if the similarity threshold is too loose?
- Which LLM responses are safe to cache?
- What metadata must be attached to cached answers before production use?

## Key Terms

| Term | Definition |
| --- | --- |
| Exact cache | Cache keyed by the exact input string |
| Semantic cache | Cache keyed by vector similarity between meanings |
| Cache hit | The answer is returned from cache instead of calling the model |
| False miss | A reusable answer is not found because the threshold is too strict |
| False hit | A cached answer is reused for a question that needed a different answer |
| TTL | Time-to-live, after which a cached entry expires |
| Document version | A version/hash of the knowledge base used when the answer was generated |

## Production Warning

Semantic caching is safest for public, stable, non-personal information such as FAQs and product docs. It is risky for personalized, private, legal, medical, financial, or fast-changing answers unless you add strong scoping and invalidation rules.
