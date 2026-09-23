# Lab 10 — Containerizing an OpenAI-Compatible API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/10_Containerization/lab10_docker.ipynb)

**Production Readiness Pack | ~45 minutes | Docker on your own machine | `OPENAI_API_KEY` to run the container**

---

## Coming from Lab 5

The Lab 5 server was started by a notebook cell and reached through ngrok. Nothing outside Colab can run it that way. This lab packages the same server as a Docker image, which is what a platform team will ask for.

**Colab can write the files but cannot build them**: it has no Docker daemon. Run the notebook anywhere, then build on a machine with Docker. Checked 2026-09-23 on Docker 29: the image builds to about 205 MB, runs as a non-root user, answers `/health`, `/v1/models`, plain and streaming chat through the OpenAI client, and returns a clean `500` when started without a key.

---

## What you will build

Four files:

1. `server.py`: Lab 5's routes, with every setting read from environment variables.
2. `requirements.txt`: pinned versions, so every build installs the same thing.
3. `.dockerignore`: keeps `.env`, notebooks and local databases out of the image.
4. `Dockerfile`: slim Python, pinned install, non-root user, `uvicorn --host 0.0.0.0`.

Then you build it, run it with the key passed in at start-up, and call it with the same OpenAI client you have used since Lab 1A.

## The idea to keep

Code and dependencies go into the image; settings and secrets arrive when the container starts. That split is what lets one image call OpenAI today and your own vLLM server tomorrow (Bonus 03) by changing `UPSTREAM_BASE_URL`. A container packages your server. It does not make a model scale.

## Key terms

| Term | Meaning |
| --- | --- |
| Image | The frozen package built from a Dockerfile |
| Container | A running instance of an image |
| Runtime secret | A value passed in when the container starts, never built into the image |
| Stateless | Keeps nothing on local disk that must survive a restart |
| Health check | A cheap endpoint the platform polls to decide whether to restart you |
| Registry | Where images are pushed so a platform can pull them |

## Next

[Lab 11 — Guardrails](../11_Guardrails_Security/README.md). No API key.
