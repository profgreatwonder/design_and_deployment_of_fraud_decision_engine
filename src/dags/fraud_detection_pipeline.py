"""
Airflow DAG for Fraud Detection Model Training Pipeline
"""

from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys

# sys.path.append('/app/src')

# --- CONFIGURATION ---
# Define absolute paths for Docker environment
AIRFLOW_HOME = "/opt/airflow"
CONFIG_PATH = os.path.join(AIRFLOW_HOME, "config/config.yaml")
MODEL_PATH = os.path.join(AIRFLOW_HOME, "src/models/fraud_detection_gru_model.h5")

# Ensure Python can find your custom modules
# If ml_training is inside your 'dags' folder, this helps Airflow find it
sys.path.append(os.path.join(AIRFLOW_HOME, "dags"))

# Default args
default_args = {
    'owner': 'fraud-detection-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'fraud_detection_training_pipeline',
    default_args=default_args,
    description='Train fraud detection GRU model',
    schedule_interval='@weekly',  # Run weekly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['fraud-detection', 'machine-learning'],
)

def preprocess_data():
    """Preprocess data"""
    from ml_training.data_preprocessor import DataPreprocessor

# Pass explicit config path
    preprocessor = DataPreprocessor(config_path=CONFIG_PATH)
    df = preprocessor.process_all()
    print(f"Preprocessing complete! Dataset shape: {df.shape}")
    return True

def train_model():
    # """Train GRU model"""
    # from ml_training.train_model import FraudDetectionTrainer
    
    # trainer = FraudDetectionTrainer()
    # model, metrics = trainer.run_training_pipeline()
    
    # print("\n=== Training Complete ===")
    # for key, value in metrics.items():
    #     print(f"{key}: {value:.4f}")
    
    # return True

    """Train GRU model"""
    from ml_training.train_model import FraudDetectionTrainer
    
    # Pass explicit config path
    trainer = FraudDetectionTrainer(config_path=CONFIG_PATH)
    model, metrics = trainer.run_training_pipeline()
    
    print("\n=== Training Complete ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    return True

def validate_model():
    # """Validate model exists and is loadable"""
    # import os
    # from tensorflow import keras
    
    # model_path = "src/models/fraud_detection_gru_model.h5"
    
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Model not found at {model_path}")
    
    # # Try to load model
    # model = keras.models.load_model(model_path)
    # print(f"✅ Model validated successfully")
    # print(f"Model summary:")
    # model.summary()
    
    # return True


    """Validate model exists and is loadable"""
    import os
    from tensorflow import keras
    
    print(f"🔍 Looking for model at: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        # List directory contents to help debugging if it fails
        parent_dir = os.path.dirname(MODEL_PATH)
        if os.path.exists(parent_dir):
            print(f"Contents of {parent_dir}: {os.listdir(parent_dir)}")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # Try to load model
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model validated successfully")
    print(f"Model summary:")
    model.summary()
    
    return True

# Task 1: Preprocess data
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

# Task 2: Train model
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# Task 3: Validate model
validate_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

# Task 4: Send notification (placeholder)
notify_task = BashOperator(
    task_id='send_notification',
    bash_command='echo "Model training completed successfully!"',
    dag=dag,
)

# Define task dependencies
preprocess_task >> train_task >> validate_task >> notify_task