#!/bin/bash

# Stop the script if any command fails
set -e

echo "🛑 Stopping existing containers..."
# Use -v if you want to wipe the database (Optional: remove -v to keep data)
docker compose -f docker-compose.yaml --profile flower down

echo "🔑 Setting permissions..."
chmod +x wait-for-it.sh
chmod +x init-multiple-dbs.sh

echo "🏗️ Building images (Sequential to avoid race conditions)..."
# 1. Build webserver first to create the base image
docker compose -f docker-compose.yaml --profile flower build airflow-webserver --no-cache
# 2. Build the rest
docker compose -f docker-compose.yaml --profile flower build --no-cache

echo "🚀 Starting Infrastructure (DB, Redis, Kafka, MinIO)..."
docker compose -f docker-compose.yaml up -d postgres redis flower zookeeper kafka kafka-ui mc minio

echo "⚡ Running Airflow Initialization..."
docker compose -f docker-compose.yaml up -d airflow-init

# Smart Wait: Loops until airflow-init finishes successfully
echo "⏳ Waiting for Airflow Init to complete..."
while [ "$(docker inspect -f '{{.State.Running}}' $(docker compose -f src/docker-compose.yaml ps -q airflow-init))" = "true" ]; do
    echo -n "."
    sleep 2
done
echo ""
echo "✅ Airflow Init complete."

echo "🚀 Starting Airflow Core Services..."
docker compose -f docker-compose.yaml --profile flower up -d airflow-webserver airflow-scheduler airflow-dag-processor airflow-triggerer airflow-cli airflow-worker mlflow-server

echo "🚀 Starting Apps (Streamlit, Producer, Consumer)..."
docker compose -f docker-compose.yaml up -d producer consumer streamlit

echo "✅ SYSTEM RESTART COMPLETE!"
echo "------------------------------------------------"
echo "Streamlit: http://$(curl -s icanhazip.com):8501"
echo "Airflow:   http://$(curl -s icanhazip.com):8080"
echo "MLflow:    http://$(curl -s icanhazip.com):5500"
echo "Flower:    http://$(curl -s icanhazip.com):5555"
echo "Kafka:    http://$(curl -s icanhazip.com):8082"