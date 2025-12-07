# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     # gcc \
#     # g++ \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy startup script
# COPY start.sh .
# RUN chmod +x start.sh

# # Expose MLflow port
# EXPOSE 5000

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:5000/health || exit 1

# # Run the startup script
# CMD ["./start.sh"]




# FROM python:3.11-slim

# # Set environment variables to reduce memory usage
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PIP_NO_CACHE_DIR=1

# # Set working directory
# WORKDIR /app

# # Install system dependencies (minimal)
# º
# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy startup script
# COPY start.sh .
# RUN chmod +x start.sh

# # Expose MLflow port
# EXPOSE 5000

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:5000/health || exit 1

# # Run the startup script with explicit shell
# # CMD ["./start.sh"]
# ENTRYPOINT ["/bin/sh", "./start.sh"]


# FROM python:3.10.16-slim-bookworm
# WORKDIR /app
# COPY requirements.txt /app/requirements.txt

# RUN pip install -r /app/requirements.txt

# COPY . .

# EXPOSE 5500


FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY src/mlflow/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5500

CMD ["mlflow", "server", "--port", "5500", "--host", "0.0.0.0"]