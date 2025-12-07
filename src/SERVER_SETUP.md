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

docker compose -f src/docker-compose.yaml --profile flower up -d --build

Copy the External IP for VM from the GCP console.

Streamlit: http://<EXTERNAL_IP>:8501

Airflow: http://<EXTERNAL_IP>:8080

MLflow: http://<EXTERNAL_IP>:5500
