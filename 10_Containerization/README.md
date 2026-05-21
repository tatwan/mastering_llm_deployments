# Lab 10 - Containerizing an OpenAI-Compatible API


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/mastering_llm_deployments/blob/main/10_Containerization/lab10_docker.ipynb)
**Production Readiness Pack | ~60 minutes | Local Docker recommended | OpenAI API key optional for build, required for live proxy**

---

## Purpose

Colab, ngrok, and Gradio share links are excellent for learning and demos. Production teams usually ask a different question:

> Can I run this service as a standard deployable unit on Cloud Run, ECS, App Runner, Kubernetes, or an internal platform?

This lab packages a Lab 5-style OpenAI-compatible FastAPI server into a Docker container.

## What You Will Build

1. `server.py` with `/health`, `/v1/models`, and `/v1/chat/completions`.
2. `requirements.txt` for the API runtime.
3. `.dockerignore` to keep secrets and notebook files out of the image.
4. A non-root `Dockerfile` with a health-check-friendly API command.
5. Local build/run commands and an OpenAI SDK smoke test.
6. A cloud deployment decision guide.

## Student Questions This Lab Answers

- What exactly goes inside a Docker image?
- How do environment variables and secrets flow into a container?
- Why must containers be stateless?
- How do I test an OpenAI-compatible API after packaging it?
- When is Docker enough, and when do I need vLLM, autoscaling, or Kubernetes?

## Key Terms

| Term | Definition |
| --- | --- |
| Image | Immutable package built from a Dockerfile |
| Container | Running instance of an image |
| Stateless service | Service that does not depend on local mutable state between requests |
| Registry | Place where images are pushed before cloud deployment |
| Health check | Endpoint used by platforms to decide if the service is alive |
| Secret | Sensitive value injected at runtime, not baked into the image |
