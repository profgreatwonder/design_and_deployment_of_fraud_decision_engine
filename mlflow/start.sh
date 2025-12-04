#!/bin/bash

set -e

echo "Starting MLflow server..."
echo "Backend store: MongoDB"
echo "Artifact store: MinIO (S3-compatible)"

# Validate required environment variables
if [ -z "$POSTGRES_URL" ]; then
    echo "ERROR: MONGODB_URI is not set"
    exit 1
fi

if [ -z "$MINIO_BUCKET" ]; then
    echo "WARNING: MINIO_BUCKET not set, using default 'mlflow'"
    export MINIO_BUCKET="mlflow"
fi

# Wait a moment for services to be ready
sleep 3

# Start MLflow server
exec mlflow server \
    --backend-store-uri "${POSTGRES_URL}" \
    --default-artifact-root "s3://${MINIO_BUCKET}/mlflow-artifacts" \
    --host 0.0.0.0 \
    --port 5000 \
    --gunicorn-opts "--timeout 120 --workers 2 --keep-alive 5"