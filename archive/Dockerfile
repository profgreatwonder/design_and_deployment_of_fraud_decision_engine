# # Dockerfile
# FROM python:3.9-slim

# LABEL maintainer="MLflow Team"
# LABEL version="2.0"
# LABEL description="Complete MLflow with MinIO and MongoDB on Render"

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     curl \
#     wget \
#     gnupg \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy application
# COPY . .

# # Create directories
# RUN mkdir -p /app/logs /app/cache

# # Create non-root user
# RUN useradd -m -u 1000 mlflowuser && \
#     chown -R mlflowuser:mlflowuser /app
# USER mlflowuser

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
#     CMD python scripts/health_check.py

# EXPOSE 5000

# CMD ["python", "mlflow/server.py"]