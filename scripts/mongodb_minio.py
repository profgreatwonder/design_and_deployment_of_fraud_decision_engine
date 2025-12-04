# # # scripts/init_all.py
# # #!/usr/bin/env python
# # """
# # Initialize both MongoDB and MinIO for MLflow
# # """
# # import subprocess
# # import sys
# # import os

# # def run_initialization():
# #     print("🚀 Starting MLflow Infrastructure Initialization")
# #     print("=" * 60)
    
# #     # Run MongoDB initialization
# #     print("\n📊 Initializing MongoDB...")
# #     mongo_result = subprocess.run(
# #         [sys.executable, "scripts/init_mongo.py"],
# #         capture_output=True,
# #         text=True
# #     )
    
# #     print(mongo_result.stdout)
# #     if mongo_result.stderr:
# #         print(f"MongoDB Errors: {mongo_result.stderr}")
    
# #     # Run MinIO initialization
# #     print("\n🪣 Initializing MinIO...")
# #     minio_result = subprocess.run(
# #         [sys.executable, "scripts/init_minio.py"],
# #         capture_output=True,
# #         text=True
# #     )
    
# #     print(minio_result.stdout)
# #     if minio_result.stderr:
# #         print(f"MinIO Errors: {minio_result.stderr}")
    
# #     # Check results
# #     print("=" * 60)
# #     if mongo_result.returncode == 0 and minio_result.returncode == 0:
# #         print("✅ All systems initialized successfully!")
# #         print("\n🎉 MLflow is ready to use!")
# #         print(f"   MLflow UI: http://localhost:5000")
# #         print(f"   MinIO Console: http://localhost:9001")
# #     else:
# #         print("❌ Initialization failed. Check errors above.")
# #         sys.exit(1)

# # if __name__ == "__main__":
# #     run_initialization()




# # scripts/init_mongo.py
# #!/usr/bin/env python
# """
# Initialize MongoDB for MLflow
# """
# import os
# import sys
# import time
# from pathlib import Path

# # Add project root
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from mlflow.config import config
# from pymongo import MongoClient, ASCENDING, DESCENDING
# from pymongo.errors import ConnectionFailure, OperationFailure


# class MongoDBInitializer:
#     """Initialize MongoDB for MLflow"""
    
#     def __init__(self):
#         self.client = None
#         self.db = None
    
#     def connect(self):
#         """Connect to MongoDB"""
#         max_retries = 10
#         retry_delay = 5
        
#         print(f"🔗 Connecting to MongoDB at {config.mongodb.mongo_host}:{config.mongodb.mongo_port}...")
        
#         for attempt in range(max_retries):
#             try:
#                 self.client = MongoClient(
#                     config.mongodb.uri,
#                     serverSelectionTimeoutMS=5000,
#                     connectTimeoutMS=10000
#                 )
                
#                 # Test connection
#                 self.client.admin.command('ping')
#                 self.db = self.client[config.mongodb.mongo_db]
                
#                 print("✅ MongoDB connection successful")
#                 return True
                
#             except ConnectionFailure as e:
#                 if attempt < max_retries - 1:
#                     print(f"Attempt {attempt + 1}/{max_retries} failed. Retrying in {retry_delay}s...")
#                     time.sleep(retry_delay)
#                 else:
#                     print(f"❌ Failed to connect to MongoDB: {e}")
#                     return False
    
#     def create_collections(self):
#         """Create MLflow collections if they don't exist"""
#         required_collections = [
#             'experiments',
#             'runs',
#             'metrics',
#             'params',
#             'tags',
#             'latest_metrics'
#         ]
        
#         existing = self.db.list_collection_names()
#         created = 0
        
#         for collection in required_collections:
#             if collection not in existing:
#                 self.db.create_collection(collection)
#                 print(f"✅ Created collection: {collection}")
#                 created += 1
#             else:
#                 print(f"📁 Collection exists: {collection}")
        
#         return created
    
#     def create_indexes(self):
#         """Create indexes for MLflow collections"""
#         indexes = {
#             'experiments': [
#                 [('experiment_id', ASCENDING), {'unique': True}],
#                 [('name', ASCENDING), {'unique': True}],
#             ],
#             'runs': [
#                 [('run_uuid', ASCENDING), {'unique': True}],
#                 [('experiment_id', ASCENDING)],
#                 [('status', ASCENDING)],
#                 [('start_time', DESCENDING)],
#                 [('end_time', DESCENDING)],
#                 [('artifact_uri', ASCENDING)],
#             ],
#             'metrics': [
#                 [('run_uuid', ASCENDING), ('key', ASCENDING), ('timestamp', ASCENDING)],
#                 [('run_uuid', ASCENDING), ('step', ASCENDING)],
#                 [('key', ASCENDING), ('value', DESCENDING)],
#             ],
#             'params': [
#                 [('run_uuid', ASCENDING), ('key', ASCENDING), {'unique': True}],
#             ],
#             'tags': [
#                 [('run_uuid', ASCENDING), ('key', ASCENDING), {'unique': True}],
#             ],
#             'latest_metrics': [
#                 [('run_uuid', ASCENDING), ('key', ASCENDING), {'unique': True}],
#             ]
#         }
        
#         created = 0
#         for collection_name, collection_indexes in indexes.items():
#             collection = self.db[collection_name]
            
#             for index_spec, index_options in collection_indexes:
#                 try:
#                     # Generate index name
#                     index_name = '_'.join([f'{field}_{direction}' for field, direction in index_spec])
                    
#                     # Check if index exists
#                     existing_indexes = list(collection.list_indexes())
#                     existing_names = [idx['name'] for idx in existing_indexes]
                    
#                     if index_name not in existing_names:
#                         collection.create_index(index_spec, **index_options)
#                         print(f"✅ Created index: {collection_name}.{index_name}")
#                         created += 1
#                     else:
#                         print(f"📊 Index exists: {collection_name}.{index_name}")
                        
#                 except OperationFailure as e:
#                     print(f"⚠️  Failed to create index {index_name} on {collection_name}: {e}")
        
#         return created
    
#     def create_admin_user(self):
#         """Create admin user for MongoDB (if needed)"""
#         try:
#             # This would only work if we're admin
#             admin_db = self.client.admin
#             users = admin_db.command('usersInfo')
            
#             if not any(user['user'] == config.mongodb.mongo_user for user in users.get('users', [])):
#                 admin_db.command(
#                     'createUser',
#                     config.mongodb.mongo_user,
#                     pwd=config.mongodb.mongo_password,
#                     roles=[
#                         {'role': 'readWrite', 'db': config.mongodb.mongo_db},
#                         {'role': 'dbAdmin', 'db': config.mongodb.mongo_db}
#                     ]
#                 )
#                 print(f"✅ Created user: {config.mongodb.mongo_user}")
#             else:
#                 print(f"👤 User exists: {config.mongodb.mongo_user}")
                
#         except Exception as e:
#             print(f"⚠️  Could not create user (may need admin privileges): {e}")
    
#     def initialize(self):
#         """Main initialization method"""
#         print("=" * 60)
#         print("📊 MongoDB Initialization for MLflow")
#         print("=" * 60)
        
#         # Connect
#         if not self.connect():
#             return False
        
#         # Create collections
#         print("\n📁 Creating collections...")
#         collections_created = self.create_collections()
        
#         # Create indexes
#         print("\n📊 Creating indexes...")
#         indexes_created = self.create_indexes()
        
#         # Try to create user (optional)
#         print("\n👤 Setting up user...")
#         self.create_admin_user()
        
#         # Close connection
#         self.client.close()
        
#         # Summary
#         print("\n" + "=" * 60)
#         print("✅ MongoDB initialized successfully!")
#         print(f"📊 Database: {config.mongodb.mongo_db}")
#         print(f"📁 Collections created: {collections_created}")
#         print(f"📈 Indexes created: {indexes_created}")
#         print(f"🔗 Connection URI: {config.mongodb.uri}")
#         print("=" * 60)
        
#         return True


# if __name__ == "__main__":
#     try:
#         initializer = MongoDBInitializer()
#         success = initializer.initialize()
#         sys.exit(0 if success else 1)
#     except Exception as e:
#         print(f"❌ Initialization failed: {e}")
#         sys.exit(1)