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





# !/bin/bash

set -e

echo "Starting MLflow server..."
echo "Backend store: PostgreSQL"
echo "Artifact store: MinIO (S3-compatible)"

# Fail fast if required vars missing
[ -z "$POSTGRES_URL" ] && echo "POSTGRES_URL is required" && exit 1
[ -z "$MINIO_BUCKET" ] && echo "MINIO_BUCKET is required" && exit 1

# Validate required environment variables
# if [ -z "$POSTGRES_URL" ]; then
#     echo "ERROR: POSTGRES_URL is not set"
#     echo "Please set PostgreSQL connection string"
#     exit 1
# fi

# if [ -z "$MINIO_BUCKET" ]; then
#     echo "WARNING: MINIO_BUCKET not set, using default 'mlflow'"
#     export MINIO_BUCKET="mlflow"
# fi

# Wait a moment for services to be ready
# sleep 3

# Start MLflow server with reduced memory footprint
exec mlflow server \
    --backend-store-uri "${POSTGRES_URL}" \
    --default-artifact-root "s3://${MINIO_BUCKET}/mlflow-artifacts" \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts
    # --gunicorn-opts "--timeout 120 --workers 1 --threads 2 --keep-alive 5 --max-requests 100 --max-requests-jitter 10"
    # --gunicorn-opts "--workers 1 --threads 1 --timeout 180 --keep-alive 2 --max-requests 50 --preload"



#!/bin/bash

# set -e

# PORT=${PORT:-5000}

# echo "Starting MLflow tracking server (single-process mode, ultra-low memory)"
# echo "Port: $PORT"
# echo "Backend: $POSTGRES_URL"
# echo "Artifacts: s3://${MINIO_BUCKET}/mlflow-artifacts/"

# # echo "Starting MLflow server..."
# # echo "Backend store: PostgreSQL"
# # echo "Artifact store: MinIO (S3-compatible)"

# [ -z "$POSTGRES_URL" ] && echo "ERROR: POSTGRES_URL missing" && exit 1
# [ -z "$MLFLOW_S3_ENDPOINT_URL" ] && echo "ERROR: MLFLOW_S3_ENDPOINT_URL missing" && exit 1

# exec mlflow server \
#     --backend-store-uri "$POSTGRES_URL" \
#     # --default-artifact-root "s3://${MINIO_BUCKET:-mlflow}/mlflow-artifacts/" \
#     --default-artifact-root "s3://${MINIO_BUCKET}/mlflow-artifacts" \
#     --host 0.0.0.0 \
#     --port "$PORT" \
#     --serve-artifacts