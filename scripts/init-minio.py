# # # scripts/init_minio.py
# # import os
# # import time
# # from minio import Minio
# # from minio.error import S3Error
# # from dotenv import load_dotenv

# # load_dotenv()

# # class MinIOInitializer:
# #     def __init__(self):
# #         self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
# #         self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
# #         self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
# #         self.bucket = os.getenv("MINIO_BUCKET", "mlflow-artifacts")
# #         self.secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        
# #     def wait_for_minio(self):
# #         """Wait for MinIO to be ready"""
# #         print(f"Waiting for MinIO at {self.endpoint}...")
        
# #         client = Minio(
# #             self.endpoint,
# #             access_key=self.access_key,
# #             secret_key=self.secret_key,
# #             secure=self.secure
# #         )
        
# #         max_retries = 10
# #         for attempt in range(max_retries):
# #             try:
# #                 # Try to list buckets (simple health check)
# #                 client.list_buckets()
# #                 print("✅ MinIO connection successful")
# #                 return client
# #             except Exception as e:
# #                 if attempt < max_retries - 1:
# #                     print(f"Attempt {attempt + 1}/{max_retries} failed. Retrying in 5s...")
# #                     time.sleep(5)
# #                 else:
# #                     print(f"❌ Failed to connect to MinIO after {max_retries} attempts")
# #                     print(f"Error: {e}")
# #                     return None
        
# #         return None
    
# #     def create_bucket_if_not_exists(self, client):
# #         """Create bucket if it doesn't exist"""
# #         try:
# #             if not client.bucket_exists(self.bucket):
# #                 client.make_bucket(self.bucket)
# #                 print(f"✅ Created bucket: {self.bucket}")
                
# #                 # Set bucket policy for MLflow (read/write)
# #                 policy = {
# #                     "Version": "2012-10-17",
# #                     "Statement": [
# #                         {
# #                             "Effect": "Allow",
# #                             "Principal": {"AWS": ["*"]},
# #                             "Action": [
# #                                 "s3:GetObject",
# #                                 "s3:PutObject",
# #                                 "s3:DeleteObject",
# #                                 "s3:ListBucket"
# #                             ],
# #                             "Resource": [
# #                                 f"arn:aws:s3:::{self.bucket}",
# #                                 f"arn:aws:s3:::{self.bucket}/*"
# #                             ]
# #                         }
# #                     ]
# #                 }
                
# #                 client.set_bucket_policy(self.bucket, policy)
# #                 print(f"✅ Set bucket policy for: {self.bucket}")
# #             else:
# #                 print(f"📦 Bucket exists: {self.bucket}")
                
# #             return True
# #         except S3Error as e:
# #             print(f"❌ Error creating bucket: {e}")
# #             return False
    
# #     def initialize(self):
# #         """Main initialization method"""
# #         print("=" * 50)
# #         print("MinIO Initialization for MLflow")
# #         print("=" * 50)
        
# #         # Connect to MinIO
# #         client = self.wait_for_minio()
# #         if not client:
# #             return False
        
# #         # Create bucket
# #         success = self.create_bucket_if_not_exists(client)
        
# #         print("\n" + "=" * 50)
# #         if success:
# #             print("✅ MinIO initialized successfully for MLflow")
# #             print(f"Bucket: {self.bucket}")
# #             print(f"Endpoint: {self.endpoint}")
# #         else:
# #             print("❌ MinIO initialization failed")
# #         print("=" * 50)
        
# #         return success

# # if __name__ == "__main__":
# #     initializer = MinIOInitializer()
# #     initializer.initialize()






# # scripts/init_minio.py
# #!/usr/bin/env python
# """
# Initialize MinIO bucket and user for MLflow
# """
# import os
# import sys
# import json
# import time
# from pathlib import Path

# # Add project root
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from mlflow.config import config
# import boto3
# from botocore.exceptions import ClientError


# class MinIOInitializer:
#     """Initialize MinIO for MLflow"""
    
#     def __init__(self):
#         self.s3_client = boto3.client('s3', **config.minio.s3_config)
#         self.s3_resource = boto3.resource('s3', **config.minio.s3_config)
        
#     def create_bucket(self):
#         """Create MLflow bucket if it doesn't exist"""
#         try:
#             # Check if bucket exists
#             self.s3_client.head_bucket(Bucket=config.minio.minio_bucket)
#             print(f"✅ Bucket '{config.minio.minio_bucket}' already exists")
#             return True
#         except ClientError as e:
#             error_code = e.response['Error']['Code']
#             if error_code == '404':
#                 # Create bucket
#                 print(f"🪣 Creating bucket '{config.minio.minio_bucket}'...")
                
#                 try:
#                     self.s3_client.create_bucket(Bucket=config.minio.minio_bucket)
#                     print(f"✅ Created bucket '{config.minio.minio_bucket}'")
                    
#                     # Configure bucket
#                     self.configure_bucket()
#                     return True
                    
#                 except ClientError as e:
#                     print(f"❌ Failed to create bucket: {e}")
#                     return False
#             else:
#                 print(f"❌ Error checking bucket: {e}")
#                 return False
    
#     def configure_bucket(self):
#         """Configure bucket settings"""
#         try:
#             # Enable versioning
#             self.s3_client.put_bucket_versioning(
#                 Bucket=config.minio.minio_bucket,
#                 VersioningConfiguration={'Status': 'Enabled'}
#             )
#             print("✅ Enabled versioning")
            
#             # Set lifecycle policy (optional)
#             try:
#                 lifecycle_policy = {
#                     'Rules': [
#                         {
#                             'ID': 'TempFiles',
#                             'Status': 'Enabled',
#                             'Filter': {'Prefix': '_temp/'},
#                             'Expiration': {'Days': 1}
#                         }
#                     ]
#                 }
#                 self.s3_client.put_bucket_lifecycle_configuration(
#                     Bucket=config.minio.minio_bucket,
#                     LifecycleConfiguration=lifecycle_policy
#                 )
#                 print("✅ Set lifecycle policy")
#             except:
#                 print("⚠️  Could not set lifecycle policy")
            
#             # Create directory structure for MLflow
#             directories = [
#                 'mlflow-artifacts/',
#                 'models/',
#                 'artifacts/',
#                 'plots/',
#                 'metrics/',
#                 'datasets/'
#             ]
            
#             for directory in directories:
#                 self.s3_client.put_object(
#                     Bucket=config.minio.minio_bucket,
#                     Key=directory,
#                     Body=b''
#                 )
            
#             print("✅ Created directory structure")
            
#         except Exception as e:
#             print(f"⚠️  Could not configure bucket: {e}")
    
#     def test_access(self):
#         """Test read/write access to bucket"""
#         try:
#             test_key = f"test_access_{int(time.time())}.txt"
#             test_content = b"MLflow MinIO access test"
            
#             # Write
#             self.s3_client.put_object(
#                 Bucket=config.minio.minio_bucket,
#                 Key=test_key,
#                 Body=test_content,
#                 ContentType='text/plain'
#             )
#             print("✅ Write test passed")
            
#             # Read
#             response = self.s3_client.get_object(
#                 Bucket=config.minio.minio_bucket,
#                 Key=test_key
#             )
#             if response['Body'].read() == test_content:
#                 print("✅ Read test passed")
            
#             # Delete
#             self.s3_client.delete_object(
#                 Bucket=config.minio.minio_bucket,
#                 Key=test_key
#             )
#             print("✅ Delete test passed")
            
#             return True
            
#         except Exception as e:
#             print(f"❌ Access test failed: {e}")
#             return False
    
#     def initialize(self):
#         """Main initialization method"""
#         print("=" * 60)
#         print("🪣 MinIO Initialization for MLflow")
#         print("=" * 60)
        
#         print(f"\n🔗 Endpoint: {config.minio.endpoint}")
#         print(f"📦 Bucket: {config.minio.minio_bucket}")
#         print(f"🔐 Access Key: {config.minio.minio_access_key[:10]}...")
        
#         # Create bucket
#         if not self.create_bucket():
#             print("\n❌ Failed to create bucket")
#             return False
        
#         # Test access
#         print("\n🔍 Testing bucket access...")
#         if not self.test_access():
#             print("❌ Access tests failed")
#             return False
        
#         print("\n" + "=" * 60)
#         print("✅ MinIO initialization completed successfully!")
#         print(f"📁 Bucket: {config.minio.minio_bucket}")
#         print(f"🔗 Console: {config.minio.console_url}")
#         print(f"🔑 Access Key: {config.minio.minio_access_key}")
#         print(f"🔐 Secret Key: {config.minio.minio_secret_key[:10]}...")
#         print("=" * 60)
        
#         return True


# if __name__ == "__main__":
#     try:
#         initializer = MinIOInitializer()
#         success = initializer.initialize()
#         sys.exit(0 if success else 1)
#     except Exception as e:
#         print(f"❌ Initialization failed: {e}")
#         sys.exit(1)