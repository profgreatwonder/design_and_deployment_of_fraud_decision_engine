*N/B: select Ubuntu LTS*
# Update packages
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Install prerequisites to allow apt to use a repository over HTTPS
sudo apt-get install -y ca-certificates curl gnupg


# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
# Install the Docker Compose Plugin
sudo apt-get install -y docker-compose-plugin

git clone https://github.com/profgreatwonder/design_and_deployment_of_fraud_decision_engine.git

cd design_and_deployment_of_fraud_decision_engine/src

vi .env

# 2. Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# 3. Add your user to docker group (avoids typing 'sudo' for every docker command)
sudo usermod -aG docker $USER

# 4. Refresh your session (exit and log back in)
exit
# (Click SSH again to reconnect)

# Run Commands
- cd design_and_deployment_of_fraud_decision_engine/src
- chmod +x wait-for-it.sh
- chmod +x init-multiple-dbs.sh
- mkdir models
- docker compose build airflow-webserver --no-cache
- docker compose --profile flower build --no-cache
- docker compose up -d postgres redis flower zookeeper kafka kafka-ui mc minio
- docker compose up airflow-init
- inside the airflow-init container terminal, we will run "airflow db init"
- docker compose --profile flower up -d airflow-webserver airflow-scheduler
- docker compose --profile flower up -d airflow-dag-processor airflow-triggerer
- docker compose --profile flower up -d airflow-cli airflow-worker
- docker compose --profile flower up -d mlflow-server
- docker compose up -d producer consumer
- docker compose up -d streamlit

Copy the External IP for VM from the GCP console using *curl -4 icanhazip.com*

- Streamlit App: http://34.133.212.143:8501	(No login)
- Airflow UI: http://34.133.212.143:8080	- User: airflow/password: airflow
- MLflow UI: http://34.133.212.143:5500	(No login)
- Flower (Celery): http://34.133.212.143:5555	(No login)
- Kafka UI: http://34.133.212.143:8082	(No login)
- Minio UI - http://34.133.212.143:9001/login User: minioadmin/password: minioadmin


docker exec -it src-postgres-1 psql -U airflow -d airflow

CREATE DATABASE mlflow;
CREATE DATABASE fraud_detection;
CREATE USER mlflow WITH PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
GRANT ALL PRIVILEGES ON DATABASE fraud_detection TO airflow;
\c mlflow
GRANT ALL ON SCHEMA public TO mlflow;
ALTER SCHEMA public OWNER TO mlflow;
\q


Check if Server is Out of Memory: docker stats --no-stream

# Fixes for Memory
Enable Swap (Quickest Stability Fix)
Even with 16GB RAM, loading multiple 1.2GB datasets in parallel (Airflow workers + Streamlit + MLflow) can spike usage. Adding Swap gives Linux a "safety net" so processes slow down instead of crashing.

- Create a 8GB swap file
      sudo fallocate -l 8G /swapfile

- Secure it
      sudo chmod 600 /swapfile

- Format as swap
      sudo mkswap /swapfile

- Enable it
      sudo swapon /swapfile

- Make it permanent (so it survives reboot)
      echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

- Verify
      free -h


# Prune the build cache (where the heavy layers live)
docker builder prune --all --force

# Prune stopped containers and dangling images
docker system prune --force

# Clean up Docker files and Images from the Server
# 1. Stop all running containers
docker kill $(docker ps -q) 2>/dev/null

# 2. Delete all containers
docker rm $(docker ps -a -q) 2>/dev/null

# 3. Delete all images
docker rmi -f $(docker images -q) 2>/dev/null

# 4. Delete all volumes (Wipes the database and minio data)
docker volume rm $(docker volume ls -q) 2>/dev/null

# 5. Remove all networks
docker network prune -f

# 6. Deep clean build cache (Reclaims disk space)
docker system prune -a --volumes --force

Verify Everything is Gone
docker ps -a       # Should show no containers
docker images      # Should show no images
docker volume ls   # Should show no volumes
df -h              # Check if your disk space (Mounted on /) has been reclaimed

DB Commands
- docker exec -it src-postgres-1 psql -U airflow -d airflow -c "\dt"