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
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Build librdkafka
RUN curl -fsSL https://github.com/confluentinc/librdkafka/archive/refs/tags/v2.8.0.tar.gz -o librdkafka.tar.gz && \
    tar -xzf librdkafka.tar.gz && \
    cd librdkafka-2.8.0 && \
    ./configure --enable-ssl --enable-sasl && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf librdkafka-2.8.0 librdkafka.tar.gz

# Create certificate directories
RUN mkdir -p /etc/ssl/certs /etc/ssl/private

# Copy certificates
# COPY aiven-certs/ca.pem /etc/ssl/certs/aiven-ca.pem
# COPY aiven-certs/service.cert /etc/ssl/certs/service.cert
# COPY aiven-certs/service.key /etc/ssl/private/service.key

# # Set proper permissions
# RUN chmod 644 /etc/ssl/certs/aiven-ca.pem && \
#     chmod 644 /etc/ssl/certs/service.cert && \
#     chmod 600 /etc/ssl/private/service.key

# Copy requirements and install
COPY src/consumer/requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install -r /app/requirements.txt

# 2. COPY SHARED LIBRARIES
# This brings the decision engine and config into the container
COPY src/decisioning_engine /app/src/decisioning_engine
COPY src/config /app/src/config

COPY src/models /app/src/models

# 3. COPY THE PRODUCER CODE
COPY src/consumer /app/src/consumer

# 4. Set PYTHONPATH so python knows /app is the root of imports
ENV PYTHONPATH="/app"

# 5. Run the script using its full path inside the container
ENTRYPOINT ["python3", "src/consumer/main.py"]