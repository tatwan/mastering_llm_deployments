# Lab 12 — Evaluation and Regression Testing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/12_Evaluation_Regression/lab12_evaluation_regression.ipynb)

**Production Readiness Pack | ~45 minutes | CPU | No API key**

---

## Coming from Labs 6, 8 and 11

RAGAS scored answers, a trace explained one failure, guards blocked some attacks. This lab is the gate that keeps all of that from sliding backwards.

---

## Why this lab exists

Every change to an LLM app fixes something and can break something else. A golden set, meaning fixed questions with the behaviour each must show and checked by plain code, catches the break before users do.

The notebook runs two configurations of a small RAG system against four golden questions. The stricter one raises the pass rate from 2 of 4 to 3 of 4, and it also **breaks** "What is QLoRA?", which used to pass. A case-by-case comparison flags it as a regression. The cause is a relevance threshold copied from Lab 11: very short questions embed weakly, and this one scores 0.33 against a gate of 0.42. You lower the gate, rerun to 4 of 4, and then measure how much margin the new threshold really has (0.03).

The system under test is a scripted stand-in, so runs are free and identical. In the Capstone you swap in your real `rag()`; nothing else changes.

## The idea to keep

Compare case by case, not in total. A pass rate can go up while something that worked breaks, and the broken case is the one users notice. Cheap deterministic checks come first; an LLM judge (Lab 6's RAGAS) is for quality that string checks cannot see.

## Key terms

| Term | Meaning |
| --- | --- |
| Golden set | Fixed questions with the behaviour each one must show |
| Regression | Something that used to pass and now fails |
| Deterministic check | A pass/fail rule with no model call, e.g. must-include / must-not-include strings |
| Margin | How far the nearest cases sit from a threshold you rely on |
| Eval harness | The code that runs the golden set and reports per-case results |

## Next

End of the Production Readiness Pack. Take a golden set back to the [Capstone](../Capstone/README.md), or keep your app online with [Bonus 07 — Hugging Face Spaces](../Bonus/07_hf_spaces_deployment.md).
