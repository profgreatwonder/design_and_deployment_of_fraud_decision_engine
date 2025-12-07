# Update packages
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Install Docker Compose plugin
sudo apt-get install -y docker-compose-plugin

# Start Docker and enable it on boot
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group (so you don't need 'sudo' for docker commands)
sudo usermod -aG docker $USER