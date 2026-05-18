# 🐳 Docker Packaging for ML Models

**Module 04 | Guide 3 of 4**

This guide covers containerizing your ML applications for consistent, reproducible deployments.

---

## Why Docker?

| Benefit | Description |
|---------|-------------|
| **Consistency** | Works the same everywhere |
| **Reproducibility** | Exact dependencies every time |
| **Isolation** | No conflicts with system packages |
| **Portability** | Deploy to any cloud provider |
| **Scalability** | Easy to replicate containers |

---

## Docker Basics

### Key Concepts

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Image                         │
│  (Blueprint - created from Dockerfile)                  │
├─────────────────────────────────────────────────────────┤
│  OS Base (python:3.10-slim)                             │
│  + Dependencies (transformers, torch, fastapi)          │
│  + Your Code (app.py, model files)                      │
│  + Configuration (environment variables)                │
└─────────────────────────────────────────────────────────┘
                        │
                        │ docker run
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   Docker Container                      │
│  (Running instance of the image)                        │
├─────────────────────────────────────────────────────────┤
│  Your FastAPI app running on port 8000                  │
│  Isolated from host system                              │
└─────────────────────────────────────────────────────────┘
```

---

## Sample Dockerfile

### Basic ML Application

```dockerfile
# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for some ML libraries)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Requirements.txt

```
transformers==4.35.0
torch==2.1.0
fastapi==0.104.0
uvicorn==0.24.0
pydantic==2.5.0
```

---

## Multi-Stage Build (Optimized)

For production, use multi-stage builds to reduce image size:

```dockerfile
# Stage 1: Build
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/dependencies -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /app/dependencies /app/dependencies
ENV PYTHONPATH=/app/dependencies

# Copy application
COPY app.py .

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## GPU Support

For GPU-enabled containers:

```dockerfile
# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run with GPU:
```bash
docker run --gpus all -p 8000:8000 my-ml-app
```

---

## Build and Run Commands

### Building

```bash
# Build image
docker build -t my-ml-app .

# Build with tag
docker build -t my-ml-app:v1.0 .

# Build with build arguments
docker build --build-arg MODEL_NAME=distilbert -t my-ml-app .
```

### Running

```bash
# Run container
docker run -p 8000:8000 my-ml-app

# Run in background
docker run -d -p 8000:8000 my-ml-app

# Run with environment variables
docker run -e MODEL_PATH=/models -p 8000:8000 my-ml-app

# Run with volume mount (for model files)
docker run -v ./models:/app/models -p 8000:8000 my-ml-app
```

---

## Docker Compose

For multi-container applications:

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=distilbert-base-uncased
      - DEVICE=cpu
    volumes:
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Add Redis for caching
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

Run with:
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Best Practices

### 1. Keep Images Small

```dockerfile
# ❌ Bad
FROM python:3.10

# ✅ Good
FROM python:3.10-slim
```

### 2. Use .dockerignore

```
# .dockerignore
__pycache__
*.pyc
.git
.env
*.md
tests/
```

### 3. Cache Dependencies

```dockerfile
# Copy requirements first (cached if unchanged)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Then copy code (changes more often)
COPY . .
```

### 4. Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1
```

### 5. Non-Root User

```dockerfile
RUN useradd -m appuser
USER appuser
```

---

## Pushing to Registry

### Docker Hub

```bash
# Login
docker login

# Tag
docker tag my-ml-app username/my-ml-app:v1.0

# Push
docker push username/my-ml-app:v1.0
```

### AWS ECR

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag my-ml-app:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/my-ml-app:latest

# Push
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/my-ml-app:latest
```

---

## Complete Example

### Project Structure

```
my-ml-app/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── models/
    └── (model files)
```

### Full Dockerfile

```dockerfile
FROM python:3.10-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="Sentiment Analysis API"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time (optional)
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('$MODEL_NAME'); \
    AutoModel.from_pretrained('$MODEL_NAME')"

# Copy application
COPY app.py .

# Create non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Image too large | Use slim base, multi-stage build |
| Build fails | Check internet access, dependency versions |
| OOM during build | Increase Docker memory limit |
| Model not found | Pre-download in Dockerfile or mount volume |
| Port not accessible | Check port mapping with `-p` |

---

## Next Steps

Continue to `04_aws_ecs_deployment.md` for cloud deployment!
