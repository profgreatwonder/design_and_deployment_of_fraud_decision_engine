COMPOSE_FILE=src/docker-compose.yaml

# Fix permissions
perms:
	chmod +x wait-for-it.sh init-multiple-dbs.sh

# Full Clean Build and Start
all: down build up

# Stop everything
down:
	docker compose -f $(COMPOSE_FILE) --profile flower down

# Wipe data (Nuclear option)
clean:
	docker compose -f $(COMPOSE_FILE) --profile flower down -v

# Build sequentially
build:
	docker compose -f $(COMPOSE_FILE) --profile flower build airflow-webserver --no-cache
	docker compose -f $(COMPOSE_FILE) --profile flower build --no-cache

# Start Infrastructure
infra:
	docker compose -f $(COMPOSE_FILE) up -d postgres redis flower zookeeper kafka kafka-ui mc minio

# Init Airflow
init:
	docker compose -f $(COMPOSE_FILE) up -d airflow-init
	@echo "Waiting 30s for init..."
	@sleep 30

# Start everything else
services:
	docker compose -f $(COMPOSE_FILE) --profile flower up -d airflow-webserver airflow-scheduler airflow-dag-processor airflow-triggerer airflow-cli airflow-worker mlflow-server
	docker compose -f $(COMPOSE_FILE) up -d producer consumer streamlit

# The master command
up: perms infra init services