# # mlflow/config.py
# import os
# import sys
# from typing import Optional
# from pydantic import BaseSettings, Field, validator
# from urllib.parse import quote_plus, urlparse


# class ServiceConfig(BaseSettings):
#     """Service discovery configuration for Render"""
#     # Render provides these environment variables
#     render_service_host: Optional[str] = Field(None, env="RENDER_SERVICE_HOST")
#     render_service_port: Optional[str] = Field(None, env="RENDER_SERVICE_PORT")
#     render: bool = Field(False, env="RENDER")
    
#     @property
#     def is_render(self) -> bool:
#         return self.render or bool(self.render_service_host)


# class MongoDBConfig(BaseSettings):
#     """MongoDB Configuration for Render"""
#     # For Render internal services
#     mongo_service_name: str = Field("mlflow-mongodb", env="MONGO_SERVICE_NAME")
#     mongo_host: str = Field("localhost", env="MONGO_HOST")
#     mongo_port: int = Field(27017, env="MONGO_PORT")
#     mongo_user: str = Field("mlflow", env="MONGO_USER")
#     mongo_password: str = Field(..., env="MONGO_PASSWORD")
#     mongo_db: str = Field("mlflow", env="MONGO_DB")
#     mongo_replica_set: Optional[str] = Field(None, env="MONGO_REPLICA_SET")
    
#     @property
#     def uri(self) -> str:
#         """Generate MongoDB URI"""
#         if self.mongo_replica_set:
#             return f"mongodb://{self.mongo_user}:{quote_plus(self.mongo_password)}@{self.mongo_host}:{self.mongo_port}/{self.mongo_db}?replicaSet={self.mongo_replica_set}"
#         return f"mongodb://{self.mongo_user}:{quote_plus(self.mongo_password)}@{self.mongo_host}:{self.mongo_port}/{self.mongo_db}"
    
#     @property
#     def mlflow_uri(self) -> str:
#         """URI for MLflow backend store"""
#         return self.uri


# class MinIOConfig(BaseSettings):
#     """MinIO Configuration for Render"""
#     # For Render internal services
#     minio_service_name: str = Field("mlflow-minio", env="MINIO_SERVICE_NAME")
#     minio_host: str = Field("localhost", env="MINIO_HOST")
#     minio_port: int = Field(9000, env="MINIO_PORT")
#     minio_console_port: int = Field(9001, env="MINIO_CONSOLE_PORT")
#     minio_access_key: str = Field(..., env="MINIO_ACCESS_KEY")
#     minio_secret_key: str = Field(..., env="MINIO_SECRET_KEY")
#     minio_bucket: str = Field("mlflow-artifacts", env="MINIO_BUCKET")
#     minio_secure: bool = Field(False, env="MINIO_SECURE")
#     minio_region: str = Field("us-east-1", env="MINIO_REGION")
    
#     @property
#     def endpoint(self) -> str:
#         """MinIO endpoint"""
#         return f"{self.minio_host}:{self.minio_port}"
    
#     @property
#     def console_url(self) -> str:
#         """MinIO console URL"""
#         return f"http://{self.minio_host}:{self.minio_console_port}"
    
#     @property
#     def artifact_uri(self) -> str:
#         """Generate MLflow artifact URI"""
#         encoded_access_key = quote_plus(self.minio_access_key)
#         encoded_secret_key = quote_plus(self.minio_secret_key)
        
#         protocol = "https" if self.minio_secure else "http"
#         endpoint_url = f"{protocol}://{self.endpoint}"
        
#         return (
#             f"s3://{self.minio_bucket}"
#             f"?endpoint_url={endpoint_url}"
#             f"&aws_access_key_id={encoded_access_key}"
#             f"&aws_secret_access_key={encoded_secret_key}"
#             f"&region_name={self.minio_region}"
#         )
    
#     @property
#     def s3_config(self) -> dict:
#         """Configuration for boto3"""
#         return {
#             "endpoint_url": f"http://{self.endpoint}",
#             "aws_access_key_id": self.minio_access_key,
#             "aws_secret_access_key": self.minio_secret_key,
#             "config": {
#                 "s3": {"addressing_style": "virtual"},
#                 "signature_version": "s3v4"
#             }
#         }


# class MLflowServerConfig(BaseSettings):
#     """MLflow Server Configuration"""
#     host: str = Field("0.0.0.0", env="MLFLOW_HOST")
#     port: int = Field(5000, env="MLFLOW_PORT")
#     workers: int = Field(2, env="MLFLOW_WORKERS")  # Reduced for Render
#     timeout: int = Field(120, env="MLFLOW_TIMEOUT")
#     serve_artifacts: bool = Field(True, env="MLFLOW_SERVE_ARTIFACTS")
#     prometheus_enabled: bool = Field(False, env="MLFLOW_PROMETHEUS_ENABLED")
    
#     # Render-specific
#     web_service_url: Optional[str] = Field(None, env="RENDER_EXTERNAL_URL")
    
#     @property
#     def external_url(self) -> str:
#         """External URL for MLflow"""
#         if self.web_service_url:
#             return self.web_service_url
#         return f"http://{self.host}:{self.port}"


# class AppConfig(BaseSettings):
#     """Application Configuration"""
#     debug: bool = Field(False, env="DEBUG")
#     log_level: str = Field("INFO", env="LOG_LEVEL")
#     python_unbuffered: bool = Field(True, env="PYTHONUNBUFFERED")
    
#     # Service configurations
#     service: ServiceConfig = ServiceConfig()
#     mongodb: MongoDBConfig = MongoDBConfig()
#     minio: MinIOConfig = MinIOConfig()
#     server: MLflowServerConfig = MLflowServerConfig()
    
#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"
    
#     def validate(self):
#         """Validate all configurations"""
#         errors = []
        
#         # Check MongoDB
#         if not self.mongodb.mongo_password:
#             errors.append("MONGO_PASSWORD is required")
        
#         # Check MinIO
#         if not self.minio.minio_access_key:
#             errors.append("MINIO_ACCESS_KEY is required")
#         if not self.minio.minio_secret_key:
#             errors.append("MINIO_SECRET_KEY is required")
        
#         if errors:
#             raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
#     def display(self, hide_secrets=True):
#         """Display configuration (safely)"""
#         import json
        
#         config_dict = {
#             "service": {
#                 "is_render": self.service.is_render,
#                 "render_service_host": self.service.render_service_host,
#             },
#             "mongodb": {
#                 "host": self.mongodb.mongo_host,
#                 "port": self.mongodb.mongo_port,
#                 "database": self.mongodb.mongo_db,
#                 "user": self.mongodb.mongo_user,
#                 "password": "***" if hide_secrets else self.mongodb.mongo_password,
#             },
#             "minio": {
#                 "host": self.minio.minio_host,
#                 "port": self.minio.minio_port,
#                 "bucket": self.minio.minio_bucket,
#                 "access_key": self.minio.minio_access_key[:8] + "..." if hide_secrets else self.minio.minio_access_key,
#                 "console_url": self.minio.console_url,
#             },
#             "server": {
#                 "host": self.server.host,
#                 "port": self.server.port,
#                 "workers": self.server.workers,
#                 "external_url": self.server.external_url,
#             },
#             "app": {
#                 "debug": self.debug,
#                 "log_level": self.log_level,
#             }
#         }
        
#         return json.dumps(config_dict, indent=2)


# # Global configuration
# config = AppConfig()