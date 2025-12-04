# # scripts/init_mongo.py
# import os
# import sys
# import time
# from pymongo import MongoClient, ASCENDING
# from pymongo.errors import ConnectionFailure, OperationFailure
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# class MongoDBInitializer:
#     def __init__(self):
#         self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
#         self.db_name = os.getenv("MONGO_DB", "mlflow")
#         self.max_retries = 5
#         self.retry_delay = 5  # seconds
        
#     def wait_for_mongodb(self):
#         """Wait for MongoDB to be ready"""
#         print("Waiting for MongoDB connection...")
        
#         for attempt in range(self.max_retries):
#             try:
#                 client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
#                 client.admin.command('ping')
#                 print("✅ MongoDB connection successful")
#                 return client
#             except ConnectionFailure as e:
#                 if attempt < self.max_retries - 1:
#                     print(f"Attempt {attempt + 1}/{self.max_retries} failed. Retrying in {self.retry_delay}s...")
#                     time.sleep(self.retry_delay)
#                 else:
#                     print(f"❌ Failed to connect to MongoDB after {self.max_retries} attempts")
#                     print(f"Error: {e}")
#                     sys.exit(1)
#         return None
    
#     def create_collections_and_indexes(self, client):
#         """Create MLflow collections and indexes"""
#         db = client[self.db_name]
        
#         # List of collections MLflow expects
#         collections = [
#             'experiments',
#             'runs', 
#             'metrics',
#             'params',
#             'tags',
#             'latest_metrics'
#         ]
        
#         created_count = 0
#         for collection_name in collections:
#             if collection_name not in db.list_collection_names():
#                 db.create_collection(collection_name)
#                 print(f"✅ Created collection: {collection_name}")
#                 created_count += 1
#             else:
#                 print(f"📁 Collection exists: {collection_name}")
        
#         # Create indexes for better performance
#         self.create_indexes(db)
        
#         return created_count
    
#     def create_indexes(self, db):
#         """Create optimal indexes for MLflow"""
#         indexes_config = {
#             'experiments': [
#                 [("experiment_id", ASCENDING), {"unique": True}],
#                 [("name", ASCENDING), {"unique": True}],
#             ],
#             'runs': [
#                 [("run_uuid", ASCENDING), {"unique": True}],
#                 [("experiment_id", ASCENDING)],
#                 [("status", ASCENDING)],
#                 [("start_time", ASCENDING)],
#                 [("end_time", ASCENDING)],
#             ],
#             'metrics': [
#                 [("run_uuid", ASCENDING), ("key", ASCENDING), ("timestamp", ASCENDING)],
#                 [("run_uuid", ASCENDING), ("step", ASCENDING)],
#             ],
#             'params': [
#                 [("run_uuid", ASCENDING), ("key", ASCENDING), {"unique": True}],
#             ],
#             'tags': [
#                 [("run_uuid", ASCENDING), ("key", ASCENDING), {"unique": True}],
#             ],
#             'latest_metrics': [
#                 [("run_uuid", ASCENDING), ("key", ASCENDING), {"unique": True}],
#             ]
#         }
        
#         created_indexes = 0
#         for collection_name, index_list in indexes_config.items():
#             collection = db[collection_name]
#             existing_indexes = list(collection.list_indexes())
#             existing_index_names = [idx['name'] for idx in existing_indexes]
            
#             for index_spec, index_options in index_list:
#                 # Generate index name
#                 index_name = "_".join([f"{field}_{direction}" for field, direction in index_spec])
                
#                 if index_name not in existing_index_names:
#                     try:
#                         collection.create_index(index_spec, **index_options)
#                         print(f"✅ Created index: {collection_name}.{index_name}")
#                         created_indexes += 1
#                     except OperationFailure as e:
#                         print(f"⚠️  Failed to create index {index_name} on {collection_name}: {e}")
#                 else:
#                     print(f"📊 Index exists: {collection_name}.{index_name}")
        
#         return created_indexes
    
#     def initialize(self):
#         """Main initialization method"""
#         print("=" * 50)
#         print("MongoDB Initialization for MLflow")
#         print("=" * 50)
        
#         # Connect to MongoDB
#         client = self.wait_for_mongodb()
        
#         # Create collections and indexes
#         collections_created = self.create_collections_and_indexes(client)
        
#         # Print summary
#         print("\n" + "=" * 50)
#         print("Initialization Summary:")
#         print(f"Database: {self.db_name}")
#         print(f"URI: {self.mongo_uri}")
#         print(f"Collections created: {collections_created}")
#         print("✅ MongoDB initialized successfully for MLflow")
#         print("=" * 50)
        
#         client.close()

# if __name__ == "__main__":
#     initializer = MongoDBInitializer()
#     initializer.initialize()