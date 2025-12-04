"""
Script to initialize MinIO bucket for MLflow
Run this once after deploying MinIO to create the required bucket
"""

import os
from minio import Minio
from minio.error import S3Error

def setup_minio_bucket():
    """Create MLflow artifacts bucket in MinIO"""
    
    # Get credentials from environment variables
    minio_endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
    minio_access_key = os.getenv('MINIO_ROOT_USER', 'minioadmin')
    minio_secret_key = os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin')
    bucket_name = os.getenv('MINIO_BUCKET', 'mlflow')
    
    # Remove http:// or https:// if present
    minio_endpoint = minio_endpoint.replace('https://', '').replace('http://', '')
    
    # Determine if we should use secure connection (https)
    secure = not minio_endpoint.startswith('localhost')
    
    print(f"Connecting to MinIO at {minio_endpoint}")
    print(f"Using secure connection: {secure}")
    
    try:
        # Initialize MinIO client
        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=secure
        )
        
        # Check if bucket exists
        if client.bucket_exists(bucket_name):
            print(f"Bucket '{bucket_name}' already exists")
        else:
            # Create bucket
            client.make_bucket(bucket_name)
            print(f"Successfully created bucket '{bucket_name}'")
        
        # Set bucket policy to allow read/write
        # This is a simple policy - adjust based on your security needs
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": [
                        "s3:GetBucketLocation",
                        "s3:ListBucket"
                    ],
                    "Resource": f"arn:aws:s3:::{bucket_name}"
                },
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject"
                    ],
                    "Resource": f"arn:aws:s3:::{bucket_name}/*"
                }
            ]
        }
        
        import json
        client.set_bucket_policy(bucket_name, json.dumps(policy))
        print(f"Successfully set policy for bucket '{bucket_name}'")
        
        print("\n✓ MinIO setup completed successfully!")
        print(f"Bucket '{bucket_name}' is ready for MLflow artifacts")
        
    except S3Error as e:
        print(f"Error occurred: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = setup_minio_bucket()
    exit(0 if success else 1)