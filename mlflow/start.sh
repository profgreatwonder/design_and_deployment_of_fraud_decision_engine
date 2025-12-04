# #!/bin/bash

# set -e

# echo "Starting MLflow server..."
# echo "Backend store: MongoDB"
# echo "Artifact store: MinIO (S3-compatible)"

# # Validate required environment variables
# if [ -z "$POSTGRES_URL" ]; then
#     echo "ERROR: POSTGRES_URL is not set"
#     exit 1
# fi

# if [ -z "$MINIO_BUCKET" ]; then
#     echo "WARNING: MINIO_BUCKET not set, using default 'mlflow'"
#     export MINIO_BUCKET="mlflow"
# fi

# # Wait a moment for services to be ready
# sleep 3

# # Start MLflow server
# exec mlflow server \
#     --backend-store-uri "${POSTGRES_URL}" \
#     --default-artifact-root "s3://${MINIO_BUCKET}/mlflow-artifacts" \
#     --host 0.0.0.0 \
#     --port 5000 \
#     --gunicorn-opts "--timeout 120 --workers 2 --keep-alive 5"





#!/bin/bash

set -e

echo "Starting MLflow server..."
echo "Backend store: PostgreSQL"
echo "Artifact store: MinIO (S3-compatible)"

# Validate required environment variables
if [ -z "$POSTGRES_URL" ]; then
    echo "ERROR: POSTGRES_URL is not set"
    echo "Please set PostgreSQL connection string"
    exit 1
fi

if [ -z "$MINIO_BUCKET" ]; then
    echo "WARNING: MINIO_BUCKET not set, using default 'mlflow'"
    export MINIO_BUCKET="mlflow"
fi

# Wait a moment for services to be ready
sleep 3

# Start MLflow server with reduced memory footprint
exec mlflow server \
    --backend-store-uri "${POSTGRES_URL}" \
    --default-artifact-root "s3://${MINIO_BUCKET}/mlflow-artifacts" \
    --host 0.0.0.0 \
    --port 5000 \
    --gunicorn-opts "--timeout 120 --workers 1 --threads 2 --keep-alive 5 --max-requests 100 --max-requests-jitter 10"