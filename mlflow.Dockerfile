FROM ghcr.io/mlflow/mlflow:v3.3.0

# psycopg2-binary untuk PostgreSQL backend, curl untuk healthcheck
RUN pip install --no-cache-dir psycopg2-binary && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
