# FROM python:3.10.16-slim-bookworm

# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONUNBUFFERED=1 \
#     PYTHONIOENCODING=UTF-8

# WORKDIR /app

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         curl \
#         gnupg \
#         gcc \
#         g++ \
#         build-essential \
#         python3-dev \
#         libssl-dev \
#         libsasl2-dev \
#         pkg-config \
#         && rm -rf /var/lib/apt/lists/*

# # manually install librdkafka with SASL/SSL support
# RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
#     curl -fsSL https://github.com/confluentinc/librdkafka/archive/refs/tags/v2.8.0.tar.gz -o librdkafka.tar.gz \
#     tar -xzf librdkafka.tar.gz && \
#     cd librdkafka-2.8.0 && \
#     ./configure --enable-ssl --enable-sasl && \
#     make && \
#     make install && \
#     ldconfig && \
#     cd .. && \
#     rm -rf librdkafka-2.8.0 librdkafka.tar.gz

# # copy requirements and install python dependencies
# COPY ./requirements.txt /app/requirements.txt
# RUN pip install -r /app/requirements.txt


# # copy application code
# COPY . .

# # entrypoint
# ENTRYPOINT ["python3", "main.py"]





FROM python:3.10.16-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        gcc \
        g++ \
        build-essential \
        python3-dev \
        libssl-dev \
        libsasl2-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

# Manually install librdkafka with SASL/SSL support
# Build args to handle proxy settings
ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy=""

RUN export http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" && \
    curl -fsSL https://github.com/confluentinc/librdkafka/archive/refs/tags/v2.8.0.tar.gz -o librdkafka.tar.gz && \
    tar -xzf librdkafka.tar.gz && \
    cd librdkafka-2.8.0 && \
    ./configure --enable-ssl --enable-sasl && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf librdkafka-2.8.0 librdkafka.tar.gz

# Copy CA certificate
# COPY ca.pem /app/ca.pem

# # Copy certificates
# COPY aiven-certs/ca.pem /etc/ssl/certs/aiven-ca.pem
# COPY aiven-certs/service.cert /etc/ssl/certs/service.cert
# COPY aiven-certs/service.key /etc/ssl/private/service.key

# # Set proper permissions
# RUN chmod 644 /etc/ssl/certs/aiven-ca.pem && \
#     chmod 644 /etc/ssl/certs/service.cert && \
#     chmod 600 /etc/ssl/private/service.key


# Copy requirements and install
COPY src/producer/requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY data /app/data

# 2. COPY SHARED LIBRARIES
# This brings the decision engine and config into the container
COPY src/decisioning_engine /app/src/decisioning_engine
COPY src/config /app/src/config

# 3. COPY THE PRODUCER CODE
COPY src/producer /app/src/producer

# 4. Set PYTHONPATH so python knows /app is the root of imports
ENV PYTHONPATH="/app"

# 5. Run the script using its full path inside the container
ENTRYPOINT ["python3", "src/producer/main.py"]