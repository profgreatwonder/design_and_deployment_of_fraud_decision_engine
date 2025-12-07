FROM apache/airflow:2.10.5

USER root
RUN mkdir -p /app/models

USER airflow
# FIX FOR TIMEOUTS ON MAC / UNSTABLE NETWORKS
ENV PIP_TIMEOUT=300 \
    PIP_RETRY=5 \
    PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org" \
    PIP_INDEX_URL="https://pypi.org/simple" \
    PIP_NO_CACHE_DIR=1

USER airflow

COPY src/airflow/requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

USER airflow




# FROM apache/airflow:2.10.5

# # ← STILL ROOT HERE
# RUN mkdir -p /app/models && \
#     chown -R airflow:0 /app

# # Now switch to the airflow user (this line already exists in the official image,
# # but keeping it explicit is fine)
# USER airflow

# # Copy your stuff if needed
# COPY --chown=airflow:0 requirements.txt /app/
# COPY --chown=airflow:0 your_script.py /app/

# WORKDIR /app


# FROM apache/airflow:2.10.5

# USER root
# RUN mkdir -p /app/models && chown -R airflow:0 /app

# USER airflow   # go back to non-root for the rest of the build / runtime