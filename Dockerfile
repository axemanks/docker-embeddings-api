# 1. Builder stage: install dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 2. Final stage: only runtime artifacts
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy installed packages, binaries, and app code
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY app/ /app/

# Use Gunicorn + Uvicorn worker for production
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8001"]
