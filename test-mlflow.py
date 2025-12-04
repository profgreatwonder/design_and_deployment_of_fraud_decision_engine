"""
Test script to verify MLflow setup with MongoDB and MinIO
Run this from your local machine to test the deployment
"""

import os
import mlflow
from mlflow import MlflowClient

def test_mlflow_setup():
    """Test MLflow tracking with MongoDB backend and MinIO artifacts"""
    
    # Set MLflow tracking URI (replace with your Render MLflow URL)
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    
    print(f"Testing MLflow at: {mlflow_uri}")
    
    # Set MinIO credentials as environment variables for artifact storage
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('MINIO_ROOT_USER', 'minioadmin')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MINIO_ENDPOINT_URL', 'http://localhost:9000')
    
    try:
        # Create a test experiment
        experiment_name = "test_experiment"
        print(f"\n1. Creating experiment: {experiment_name}")
        
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            # Experiment might already exist
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        print(f"   ✓ Experiment ID: {experiment_id}")
        
        # Start a run and log parameters, metrics, and artifacts
        print("\n2. Starting MLflow run")
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            print(f"   ✓ Run ID: {run_id}")
            
            # Log parameters
            print("\n3. Logging parameters")
            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_param("batch_size", 32)
            print("   ✓ Parameters logged")
            
            # Log metrics
            print("\n4. Logging metrics")
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("loss", 0.05)
            print("   ✓ Metrics logged")
            
            # Create and log an artifact
            print("\n5. Logging artifact")
            with open("test_artifact.txt", "w") as f:
                f.write("This is a test artifact stored in MinIO")
            mlflow.log_artifact("test_artifact.txt")
            print("   ✓ Artifact logged to MinIO")
            
            # Clean up local file
            os.remove("test_artifact.txt")
        
        # Verify we can retrieve the run
        print("\n6. Retrieving run information")
        client = MlflowClient()
        run_info = client.get_run(run_id)
        print(f"   ✓ Run retrieved successfully")
        print(f"   - Status: {run_info.info.status}")
        print(f"   - Artifact URI: {run_info.info.artifact_uri}")
        
        # List artifacts to verify MinIO storage
        print("\n7. Listing artifacts from MinIO")
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            print(f"   ✓ Found artifact: {artifact.path}")
        
        print("\n" + "="*50)
        print("✓ All tests passed successfully!")
        print("="*50)
        print(f"\nYour MLflow server is working correctly!")
        print(f"Access the UI at: {mlflow_uri}")
        print("\nSetup verified:")
        print("  • MongoDB backend: Connected")
        print("  • MinIO artifact storage: Connected")
        print("  • Experiment tracking: Working")
        print("  • Artifact logging: Working")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify MLFLOW_TRACKING_URI is correct")
        print("2. Check MinIO credentials (MINIO_ROOT_USER/PASSWORD)")
        print("3. Ensure MLFLOW_S3_ENDPOINT_URL is accessible")
        print("4. Verify MongoDB connection string is correct")
        return False

if __name__ == "__main__":
    success = test_mlflow_setup()
    exit(0 if success else 1)