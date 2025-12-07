#!/bin/sh

# Start MinIO server
# For Render, we bind to 0.0.0.0:9000 (Render maps to port 10000 externally)
# MINIO_ROOT_USER and MINIO_ROOT_PASSWORD must be set as environment variables in Render

exec minio server /data --address "0.0.0.0:9000" --console-address ":9001"