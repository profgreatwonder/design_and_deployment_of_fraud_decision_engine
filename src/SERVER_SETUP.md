# Update packages
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Update existing list
sudo apt-get update

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

git clone https://github.com/profgreatwonder/design_and_deployment_of_fraud_decision_engine.git

cd design_and_deployment_of_fraud_decision_engine

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
- chmod +x wait-for-it.sh
- chmod +x init-multiple-dbs.sh
- mkdir src/models
- docker compose build airflow-webserver --no-cache
- docker compose --profile flower build --no-cache
- docker compose up -d postgres redis flower zookeeper kafka kafka-ui mc minio
- docker compose up -d airflow-init
- sleep 30
- docker compose --profile flower up -d airflow-webserver airflow-scheduler airflow-dag-processor airflow-triggerer airflow-cli airflow-worker mlflow-server
# docker compose up -d producer consumer streamlit

Copy the External IP for VM from the GCP console using *curl -4 icanhazip.com*

- Streamlit App: http://104.154.232.45:8501	(No login)
- Airflow UI: http://104.154.232.45:8080	- User: airflow/password: airflow
- MLflow UI: http://104.154.232.45:5500	(No login)
- Flower (Celery): http://104.154.232.45:5555	(No login)
- Kafka UI: http://104.154.232.45:8082	(No login)
- Minio UI - http://104.154.232.45:9001/login User: minioadmin/password: minioadmin


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