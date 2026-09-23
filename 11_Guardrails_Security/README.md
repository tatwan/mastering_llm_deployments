# Lab 11 — Guardrails and Deployment Security

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/11_Guardrails_Security/lab11_guardrails_security.ipynb)

**Production Readiness Pack | ~45 minutes | CPU | No API key**

---

## Coming from Labs 2 and 7

A classmate attacked your app in Lab 7, and Lab 2 measured how often injection works. This lab asks what you put in front of the model before real users arrive.

---

## Why this lab exists

No single check stops everything, so production systems layer cheap checks at different points in a request. You build three with plain Python:

1. **Input guard**: patterns that usually mean "ignore your instructions".
2. **Retrieval confidence gate**: decline when nothing relevant was found.
3. **Output guard**: redact emails, phone numbers and key-shaped strings before the answer leaves.

Then you break them. Four probes produce four mistakes: two injections that slip past (one because of the single word "the"), and two innocent questions that are refused. The redacted answer still names the person whose contact details it hid. Knowing where your guards fail is the lesson.

The responder is a scripted stand-in, not a model, so the lab is free and repeatable. It obeys every injection, which makes each guard's effect easy to see; a real model obeys sometimes.

## The idea to keep

Every guardrail trades one kind of mistake for another, so treat guardrails as product decisions and measure them: block rate, false positives, unfinished user tasks. Keep sensitive data out of the index in the first place; redaction is the last line of defence, not the first.

## Key terms

| Term | Meaning |
| --- | --- |
| Prompt injection | Input that tries to override the system's instructions |
| Guardrail | A programmatic check before or after the model call |
| Retrieval gate | A minimum relevance score before generation is allowed |
| Output redaction | Removing sensitive text from an answer before it is returned |
| False positive / negative | Blocking a legitimate request / allowing a harmful one |
| Defence in depth | Several imperfect controls layered together |
| Residual risk | What is still possible after the controls are in place |

## Next

[Lab 12 — Evaluation and Regression](../12_Evaluation_Regression/README.md).
