# # mlflow/server.py
# import json
# import os
# import sys
# import subprocess
# import logging
# import time
# from pathlib import Path
# from urllib.parse import urlparse

# # Add project root to path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from mlflow.config import config
# from scripts.service_discovery import discover_services, wait_for_service

# # Configure logging
# def setup_logging():
#     log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
#     if config.debug:
#         level = logging.DEBUG
#         log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
#     else:
#         level = getattr(logging, config.log_level.upper())
    
#     logging.basicConfig(
#         level=level,
#         format=log_format,
#         handlers=[
#             logging.StreamHandler(sys.stdout),
#             logging.FileHandler('/app/logs/mlflow_server.log')
#         ]
#     )
    
#     # Reduce noisy logs
#     logging.getLogger('botocore').setLevel(logging.WARNING)
#     logging.getLogger('boto3').setLevel(logging.WARNING)
#     logging.getLogger('urllib3').setLevel(logging.WARNING)

# setup_logging()
# logger = logging.getLogger(__name__)


# class ServiceValidator:
#     """Validate and wait for dependent services"""
    
#     @staticmethod
#     def validate_mongodb():
#         """Validate MongoDB connection"""
#         from pymongo import MongoClient
#         from pymongo.errors import ConnectionFailure, OperationFailure
        
#         max_retries = 10
#         retry_delay = 5
        
#         logger.info(f"🔍 Validating MongoDB at {config.mongodb.mongo_host}:{config.mongodb.mongo_port}")
        
#         for attempt in range(max_retries):
#             try:
#                 # Connect with short timeout
#                 client = MongoClient(
#                     config.mongodb.uri,
#                     serverSelectionTimeoutMS=5000,
#                     connectTimeoutMS=10000,
#                     socketTimeoutMS=10000
#                 )
                
#                 # Test connection
#                 client.admin.command('ping')
                
#                 # Check if database exists or can be created
#                 db = client[config.mongodb.mongo_db]
#                 collections = db.list_collection_names()
                
#                 logger.info(f"✅ MongoDB connected successfully")
#                 logger.info(f"📁 Database: {config.mongodb.mongo_db}")
#                 logger.info(f"📄 Collections: {len(collections)} found")
                
#                 # Create MLflow collections if they don't exist
#                 ServiceValidator.ensure_mlflow_collections(db)
                
#                 client.close()
#                 return True
                
#             except (ConnectionFailure, OperationFailure) as e:
#                 if attempt < max_retries - 1:
#                     logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
#                     time.sleep(retry_delay)
#                 else:
#                     logger.error(f"❌ Failed to connect to MongoDB after {max_retries} attempts: {e}")
#                     return False
#             except Exception as e:
#                 logger.error(f"❌ Unexpected MongoDB error: {e}")
#                 return False
        
#         return False
    
#     @staticmethod
#     def ensure_mlflow_collections(db):
#         """Ensure MLflow collections exist"""
#         collections = ['experiments', 'runs', 'metrics', 'params', 'tags', 'latest_metrics']
        
#         for collection in collections:
#             if collection not in db.list_collection_names():
#                 db.create_collection(collection)
#                 logger.debug(f"Created collection: {collection}")
    
#     @staticmethod
#     def validate_minio():
#         """Validate MinIO connection and bucket"""
#         import boto3
#         from botocore.exceptions import ClientError
        
#         max_retries = 10
#         retry_delay = 5
        
#         logger.info(f"🔍 Validating MinIO at {config.minio.endpoint}")
        
#         for attempt in range(max_retries):
#             try:
#                 s3_client = boto3.client('s3', **config.minio.s3_config)
                
#                 # Test connection by listing buckets
#                 response = s3_client.list_buckets()
#                 buckets = [b['Name'] for b in response.get('Buckets', [])]
#                 logger.debug(f"Available buckets: {buckets}")
                
#                 # Check if our bucket exists
#                 if config.minio.minio_bucket in buckets:
#                     logger.info(f"✅ MinIO bucket '{config.minio.minio_bucket}' found")
                    
#                     # Test write permissions
#                     test_key = f"_mlflow_test/{int(time.time())}.txt"
#                     s3_client.put_object(
#                         Bucket=config.minio.minio_bucket,
#                         Key=test_key,
#                         Body=b"MLflow connection test",
#                         ContentType='text/plain'
#                     )
                    
#                     # Test read
#                     s3_client.get_object(Bucket=config.minio.minio_bucket, Key=test_key)
                    
#                     # Cleanup
#                     s3_client.delete_object(Bucket=config.minio.minio_bucket, Key=test_key)
                    
#                     logger.info("✅ MinIO read/write permissions verified")
#                     return True
#                 else:
#                     # Try to create bucket
#                     logger.info(f"Creating bucket '{config.minio.minio_bucket}'...")
#                     s3_client.create_bucket(Bucket=config.minio.minio_bucket)
                    
#                     # Set public read for artifacts (optional)
#                     try:
#                         policy = {
#                             "Version": "2012-10-17",
#                             "Statement": [{
#                                 "Effect": "Allow",
#                                 "Principal": {"AWS": ["*"]},
#                                 "Action": [
#                                     "s3:GetObject",
#                                     "s3:PutObject",
#                                     "s3:DeleteObject",
#                                     "s3:ListBucket"
#                                 ],
#                                 "Resource": [
#                                     f"arn:aws:s3:::{config.minio.minio_bucket}",
#                                     f"arn:aws:s3:::{config.minio.minio_bucket}/*"
#                                 ]
#                             }]
#                         }
#                         s3_client.put_bucket_policy(
#                             Bucket=config.minio.minio_bucket,
#                             Policy=json.dumps(policy)
#                         )
#                     except:
#                         logger.warning("Could not set bucket policy (may require admin permissions)")
                    
#                     logger.info(f"✅ Created bucket '{config.minio.minio_bucket}'")
#                     return True
                    
#             except ClientError as e:
#                 error_code = e.response.get('Error', {}).get('Code', 'Unknown')
#                 if attempt < max_retries - 1:
#                     logger.warning(f"Attempt {attempt + 1}/{max_retries} failed ({error_code}). Retrying in {retry_delay}s...")
#                     time.sleep(retry_delay)
#                 else:
#                     logger.error(f"❌ Failed to connect to MinIO after {max_retries} attempts: {e}")
#                     return False
#             except Exception as e:
#                 logger.error(f"❌ Unexpected MinIO error: {e}")
#                 return False
        
#         return False


# def build_gunicorn_command():
#     """Build the production gunicorn command"""
    
#     cmd = [
#         "gunicorn",
#         "--bind", f"{config.server.host}:{config.server.port}",
#         "--workers", str(config.server.workers),
#         "--worker-class", "sync",
#         "--timeout", str(config.server.timeout),
#         "--access-logfile", "-",
#         "--error-logfile", "-",
#         "--log-level", config.log_level.lower(),
#         "--preload",  # Preload app for faster worker startup
#         "mlflow.server:app",
#         "--",
#         "--backend-store-uri", config.mongodb.mlflow_uri,
#         "--default-artifact-root", config.minio.artifact_uri,
#         "--host", config.server.host,
#         "--port", str(config.server.port),
#     ]
    
#     # Add optional features
#     if config.server.serve_artifacts:
#         cmd.append("--serve-artifacts")
    
#     if config.server.prometheus_enabled:
#         cmd.append("--enable-mlflow-prometheus-metrics")
    
#     # Development mode
#     if config.debug:
#         cmd.append("--dev")
#         logger.warning("⚠️ Running in DEBUG mode - not for production!")
    
#     return cmd


# def start_server():
#     """Start the MLflow server with full validation"""
    
#     logger.info("=" * 70)
#     logger.info("🚀 MLflow Production Server - Complete Render Deployment")
#     logger.info("=" * 70)
    
#     # Validate configuration
#     try:
#         config.validate()
#         logger.info("✅ Configuration validated")
#     except ValueError as e:
#         logger.error(f"❌ Configuration error: {e}")
#         sys.exit(1)
    
#     # Display safe configuration
#     logger.info("\n📋 Configuration Summary:")
#     logger.info(config.display(hide_secrets=True))
    
#     # Service discovery on Render
#     if config.service.is_render:
#         logger.info("\n🔍 Discovering Render services...")
#         discover_services()
    
#     # Validate dependencies
#     logger.info("\n🔌 Validating service dependencies...")
    
#     if not ServiceValidator.validate_mongodb():
#         logger.error("❌ MongoDB validation failed. Exiting.")
#         sys.exit(1)
    
#     if not ServiceValidator.validate_minio():
#         logger.error("❌ MinIO validation failed. Exiting.")
#         sys.exit(1)
    
#     # Set environment variables for MLflow
#     os.environ.update({
#         "AWS_ACCESS_KEY_ID": config.minio.minio_access_key,
#         "AWS_SECRET_ACCESS_KEY": config.minio.minio_secret_key,
#         "AWS_DEFAULT_REGION": config.minio.minio_region,
#         "MLFLOW_TRACKING_URI": f"http://{config.server.host}:{config.server.port}",
#     })
    
#     # Build and start server
#     cmd = build_gunicorn_command()
    
#     logger.info("\n⚡ Starting MLflow server...")
#     logger.info(f"🌐 External URL: {config.server.external_url}")
#     logger.info(f"🗄️  Artifact Store: MinIO ({config.minio.endpoint})")
#     logger.info(f"💾 Backend Store: MongoDB ({config.mongodb.mongo_host})")
#     logger.info("=" * 70)
    
#     try:
#         # Log the command (without sensitive info)
#         safe_cmd = [c if not any(secret in c for secret in [
#             config.minio.minio_access_key,
#             config.minio.minio_secret_key,
#             config.mongodb.mongo_password
#         ]) else "***" for c in cmd]
#         logger.debug(f"Command: {' '.join(safe_cmd)}")
        
#         # Start the server
#         subprocess.run(cmd)
        
#     except KeyboardInterrupt:
#         logger.info("\n👋 Server shutdown requested")
#     except Exception as e:
#         logger.error(f"❌ Server crashed: {e}", exc_info=config.debug)
#         sys.exit(1)


# if __name__ == "__main__":
#     start_server()