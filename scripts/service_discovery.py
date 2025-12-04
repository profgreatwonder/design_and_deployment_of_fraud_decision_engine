# # scripts/service_discovery.py
# """
# Service discovery for Render internal services
# """
# import os
# import socket
# import time
# import logging
# from typing import Optional

# logger = logging.getLogger(__name__)


# def get_service_host(service_name: str, default_host: str = "localhost") -> str:
#     """
#     Get service hostname in Render.
#     Render uses <service-name>.<network> for internal DNS.
#     """
#     if os.getenv("RENDER"):
#         # In Render, services are accessible via service name
#         network = os.getenv("RENDER_NETWORK", "internal")
#         return f"{service_name}.{network}"
#     return default_host


# def wait_for_service(host: str, port: int, timeout: int = 60) -> bool:
#     """
#     Wait for a service to become available
#     """
#     logger.info(f"⏳ Waiting for {host}:{port}...")
    
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         try:
#             sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             sock.settimeout(5)
#             result = sock.connect_ex((host, port))
#             sock.close()
            
#             if result == 0:
#                 logger.info(f"✅ {host}:{port} is available")
#                 return True
#         except socket.error:
#             pass
        
#         time.sleep(2)
#         if int(time.time() - start_time) % 10 == 0:
#             logger.info(f"Still waiting for {host}:{port}...")
    
#     logger.error(f"❌ Timeout waiting for {host}:{port}")
#     return False


# def discover_services():
#     """
#     Discover and configure services in Render environment
#     """
#     from mlflow.config import config
    
#     logger.info("Discovering services in Render environment...")
    
#     # MongoDB discovery
#     mongo_host = get_service_host(config.mongodb.mongo_service_name, "localhost")
#     if mongo_host != config.mongodb.mongo_host:
#         logger.info(f"Discovered MongoDB at: {mongo_host}")
#         # Update config (in memory)
#         config.mongodb.mongo_host = mongo_host
    
#     # MinIO discovery
#     minio_host = get_service_host(config.minio.minio_service_name, "localhost")
#     if minio_host != config.minio.minio_host:
#         logger.info(f"Discovered MinIO at: {minio_host}")
#         # Update config (in memory)
#         config.minio.minio_host = minio_host
    
#     # Wait for services
#     services_to_wait = [
#         (config.mongodb.mongo_host, config.mongodb.mongo_port, "MongoDB"),
#         (config.minio.minio_host, config.minio.minio_port, "MinIO"),
#     ]
    
#     for host, port, name in services_to_wait:
#         if not wait_for_service(host, port, timeout=120):
#             logger.warning(f"Service {name} at {host}:{port} not ready, but continuing...")
    
#     logger.info("✅ Service discovery complete")


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     discover_services()