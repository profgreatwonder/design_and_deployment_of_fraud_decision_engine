# FROM apache/airflow:2.10.5

# USER root
# RUN mkdir -p /app/models

# USER airflow
# # FIX FOR TIMEOUTS ON MAC / UNSTABLE NETWORKS
# ENV PIP_TIMEOUT=300 \
#     PIP_RETRY=5 \
#     PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org" \
#     PIP_INDEX_URL="https://pypi.org/simple" \
#     PIP_NO_CACHE_DIR=1

# USER airflow

# COPY src/airflow/requirements.txt /tmp/requirements.txt

# RUN pip install -r /tmp/requirements.txt

# USER airflow


FROM apache/airflow:2.10.5

# 1. Switch to ROOT to install packages and create folders
USER root

# Create the directory structure Airflow needs to save models
# We assign ownership to 'airflow' user immediately
RUN mkdir -p /opt/airflow/src/models && \
    mkdir -p /opt/airflow/data && \
    chown -R airflow:0 /opt/airflow/src && \
    chown -R airflow:0 /opt/airflow/data

# 2. Switch back to AIRFLOW user for python installation
USER airflow

# 3. Configure Pip for reliability
ENV PIP_TIMEOUT=300 \
    PIP_RETRY=5 \
    PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org" \
    PIP_INDEX_URL="https://pypi.org/simple" \
    PIP_NO_CACHE_DIR=1

# 4. Install Python Requirements
COPY src/airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# (Optional) Copy shared code if you want it baked into the image
# This acts as a fallback if volumes fail
COPY --chown=airflow:0 src/decisioning_engine /opt/airflow/dags/decisioning_engine
COPY --chown=airflow:0 src/config /opt/airflow/config

# 5. Final User switch (Best Practice)
USER airflow