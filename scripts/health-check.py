# # scripts/health_check.py
# #!/usr/bin/env python
# """
# Health check for MLflow server
# """
# import sys
# import requests
# import logging
# from pathlib import Path

# # Add project root
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from mlflow.config import config

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def check_mlflow():
#     """Check MLflow server health"""
#     try:
#         url = f"http://{config.server.host}:{config.server.port}/health"
#         response = requests.get(url, timeout=5)
        
#         if response.status_code == 200:
#             logger.info("✅ MLflow server is healthy")
#             return True
#         else:
#             logger.error(f"❌ MLflow server returned {response.status_code}")
#             return False
            
#     except requests.exceptions.RequestException as e:
#         logger.error(f"❌ Cannot connect to MLflow server: {e}")
#         return False


# def check_mongodb():
#     """Check MongoDB connection"""
#     try:
#         from pymongo import MongoClient
#         from pymongo.errors import ConnectionFailure
        
#         client = MongoClient(config.mongodb.uri, serverSelectionTimeoutMS=3000)
#         client.admin.command('ping')
#         client.close()
        
#         logger.info("✅ MongoDB is healthy")
#         return True
        
#     except ConnectionFailure as e:
#         logger.error(f"❌ MongoDB connection failed: {e}")
#         return False


# def check_minio():
#     """Check MinIO connection"""
#     try:
#         import boto3
#         from botocore.exceptions import ClientError
        
#         s3 = boto3.client('s3', **config.minio.s3_config)
#         s3.list_buckets()
        
#         # Check our specific bucket
#         try:
#             s3.head_bucket(Bucket=config.minio.minio_bucket)
#             logger.info("✅ MinIO is healthy")
#             return True
#         except ClientError as e:
#             logger.error(f"❌ MinIO bucket error: {e}")
#             return False
            
#     except Exception as e:
#         logger.error(f"❌ MinIO connection failed: {e}")
#         return False


# def main():
#     """Run all health checks"""
#     logger.info("🏥 Running health checks...")
    
#     checks = [
#         ("MLflow Server", check_mlflow),
#         ("MongoDB", check_mongodb),
#         ("MinIO", check_minio),
#     ]
    
#     results = []
#     for service_name, check_func in checks:
#         logger.info(f"🔍 Checking {service_name}...")
#         result = check_func()
#         results.append((service_name, result))
    
#     # Summary
#     logger.info("\n" + "=" * 50)
#     logger.info("Health Check Summary:")
    
#     all_healthy = True
#     for service_name, is_healthy in results:
#         status = "✅ HEALTHY" if is_healthy else "❌ UNHEALTHY"
#         logger.info(f"  {service_name}: {status}")
#         if not is_healthy:
#             all_healthy = False
    
#     logger.info("=" * 50)
    
#     return 0 if all_healthy else 1


# if __name__ == "__main__":
#     sys.exit(main())