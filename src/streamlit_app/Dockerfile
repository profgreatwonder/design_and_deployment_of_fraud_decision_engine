# FROM python:3.10-slim

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         gcc \
#         g++ \
#         python3-dev \
#         libssl-dev \
#         && rm -rf /var/lib/apt/lists/*

# # Copy requirements
# COPY src/streamlit_app/requirements.txt /app/requirements.txt
# RUN pip install -r /app/requirements.txt

# # Copy Shared Libraries
# COPY src/decisioning_engine /app/src/decisioning_engine
# COPY src/config /app/src/config

# COPY src/models /app/src/models
# COPY src/data /app/data

# # Copy Streamlit Code
# COPY src/streamlit_app /app/src/streamlit_app

# ENV PYTHONPATH="/app"

# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT ["streamlit", "run", "src/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]




FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies (Cached layer)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ python3-dev libssl-dev curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies (Cached layer)
COPY src/streamlit_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3. Copy Shared Resources
COPY data /app/data
# Code expects "src/models/..." so we put it at /app/src/models
COPY src/models /app/src/models
COPY src/config /app/src/config
COPY src/decisioning_engine /app/src/decisioning_engine

# 4. Copy Streamlit App Code
COPY src/streamlit_app /app/src/streamlit_app

# 5. Set Python Path so "from src..." imports work
ENV PYTHONPATH="/app"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]