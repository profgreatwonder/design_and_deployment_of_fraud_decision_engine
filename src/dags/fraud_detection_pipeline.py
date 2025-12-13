# """
# Airflow DAG for Fraud Detection Model Training Pipeline
# """

# from datetime import datetime, timedelta
# import os
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.operators.bash import BashOperator
# import sys

# # sys.path.append('/app/src')

# # --- CONFIGURATION ---
# # Define absolute paths for Docker environment
# AIRFLOW_HOME = "/opt/airflow"
# CONFIG_PATH = os.path.join(AIRFLOW_HOME, "config/config.yaml")
# MODEL_PATH = os.path.join(AIRFLOW_HOME, "src/models/fraud_detection_gru_model.h5")

# # Ensure Python can find your custom modules
# # If ml_training is inside your 'dags' folder, this helps Airflow find it
# sys.path.append(os.path.join(AIRFLOW_HOME, "dags"))

# # Default args
# default_args = {
#     'owner': 'fraud-detection-team',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# # DAG definition
# dag = DAG(
#     'fraud_detection_training_pipeline',
#     default_args=default_args,
#     description='Train fraud detection GRU model',
#     schedule_interval='@weekly',  # Run weekly
#     start_date=datetime(2024, 1, 1),
#     catchup=False,
#     tags=['fraud-detection', 'machine-learning'],
# )

# def preprocess_data():
#     """Preprocess data"""
#     from ml_training.data_preprocessor import DataPreprocessor

# # Pass explicit config path
#     preprocessor = DataPreprocessor(config_path=CONFIG_PATH)
#     df = preprocessor.process_all()
#     print(f"Preprocessing complete! Dataset shape: {df.shape}")
#     return True

# def train_model():
#     # """Train GRU model"""
#     # from ml_training.train_model import FraudDetectionTrainer
    
#     # trainer = FraudDetectionTrainer()
#     # model, metrics = trainer.run_training_pipeline()
    
#     # print("\n=== Training Complete ===")
#     # for key, value in metrics.items():
#     #     print(f"{key}: {value:.4f}")
    
#     # return True

#     """Train GRU model"""
#     from ml_training.train_model import FraudDetectionTrainer
    
#     # Pass explicit config path
#     trainer = FraudDetectionTrainer(config_path=CONFIG_PATH)
#     model, metrics = trainer.run_training_pipeline()
    
#     print("\n=== Training Complete ===")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")
    
#     return True

# def validate_model():
#     # """Validate model exists and is loadable"""
#     # import os
#     # from tensorflow import keras
    
#     # model_path = "src/models/fraud_detection_gru_model.h5"
    
#     # if not os.path.exists(model_path):
#     #     raise FileNotFoundError(f"Model not found at {model_path}")
    
#     # # Try to load model
#     # model = keras.models.load_model(model_path)
#     # print(f"✅ Model validated successfully")
#     # print(f"Model summary:")
#     # model.summary()
    
#     # return True


#     """Validate model exists and is loadable"""
#     import os
#     from tensorflow import keras
    
#     print(f"🔍 Looking for model at: {MODEL_PATH}")
    
#     if not os.path.exists(MODEL_PATH):
#         # List directory contents to help debugging if it fails
#         parent_dir = os.path.dirname(MODEL_PATH)
#         if os.path.exists(parent_dir):
#             print(f"Contents of {parent_dir}: {os.listdir(parent_dir)}")
#         raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
#     # Try to load model
#     model = keras.models.load_model(MODEL_PATH)
#     print(f"✅ Model validated successfully")
#     print(f"Model summary:")
#     model.summary()
    
#     return True

# # Task 1: Preprocess data
# preprocess_task = PythonOperator(
#     task_id='preprocess_data',
#     python_callable=preprocess_data,
#     dag=dag,
# )

# # Task 2: Train model
# train_task = PythonOperator(
#     task_id='train_model',
#     python_callable=train_model,
#     dag=dag,
# )

# # Task 3: Validate model
# validate_task = PythonOperator(
#     task_id='validate_model',
#     python_callable=validate_model,
#     dag=dag,
# )

# # Task 4: Send notification (placeholder)
# notify_task = BashOperator(
#     task_id='send_notification',
#     bash_command='echo "Model training completed successfully!"',
#     dag=dag,
# )

# # Define task dependencies
# preprocess_task >> train_task >> validate_task >> notify_task


# WORKS

# """
# Airflow DAG for Fraud Detection Model Training Pipeline
# Using Isolation Forest
# """

# from datetime import datetime, timedelta
# import os
# import sys
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.operators.bash import BashOperator

# # --- CONFIGURATION ---
# AIRFLOW_HOME = "/opt/airflow"
# # Ensure we use the absolute path that matches your Docker volume mapping
# CONFIG_PATH = os.path.join(AIRFLOW_HOME, "config/config.yaml")
# # The consumer looks for the model here
# MODEL_PATH = os.path.join(AIRFLOW_HOME, "src/models/fraud_detection_model.pkl") 

# # Ensure Python can find your custom modules
# sys.path.append(os.path.join(AIRFLOW_HOME, "dags"))

# # Default args
# default_args = {
#     'owner': 'fraud-detection-team',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# # DAG definition
# dag = DAG(
#     'fraud_detection_training_pipeline', # Renamed for clarity
#     default_args=default_args,
#     description='Train fraud detection Isolation Forest model',
#     schedule_interval='@daily',
#     start_date=datetime(2024, 1, 1),
#     catchup=False,
#     tags=['fraud-detection', 'isolation-forest'],
# )

# def preprocess_data():
#     """Preprocess data"""
#     # Import inside function to avoid top-level import errors in Airflow
#     from ml_training.data_preprocessor import DataPreprocessor
    
#     preprocessor = DataPreprocessor(config_path=CONFIG_PATH)
#     df = preprocessor.process_all()
#     print(f"✅ Preprocessing complete! Dataset shape: {df.shape}")
#     return True

# def train_model():
#     """Train Isolation Forest model"""
#     # UPDATED: Import from train_model, not train_isolation_forest
#     from ml_training.train_model import IsolationForestTrainer
    
#     trainer = IsolationForestTrainer(config_path=CONFIG_PATH)
#     model, metrics = trainer.run_training_pipeline()
    
#     print("\n=== Training Complete ===")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
#     return True

# def validate_model():
#     """Validate model exists and is loadable"""
#     import pickle
    
#     print(f"🔍 Looking for model at: {MODEL_PATH}")
    
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
#     # Try to load model to ensure it's not corrupted
#     with open(MODEL_PATH, 'rb') as f:
#         model = pickle.load(f)
    
#     print(f"✅ Model validated successfully")
#     return True

# # Task 1: Preprocess data
# preprocess_task = PythonOperator(
#     task_id='preprocess_data',
#     python_callable=preprocess_data,
#     dag=dag,
# )

# # Task 2: Train model
# train_task = PythonOperator(
#     task_id='train_model',
#     python_callable=train_model,
#     dag=dag,
# )

# # Task 3: Validate model
# validate_task = PythonOperator(
#     task_id='validate_model',
#     python_callable=validate_model,
#     dag=dag,
# )

# # Task 4: Notification
# notify_task = BashOperator(
#     task_id='send_notification',
#     bash_command='echo "✅ Isolation Forest model training completed successfully!"',
#     dag=dag,
# )

# # Define task dependencies
# preprocess_task >> train_task >> validate_task >> notify_task


# CLAUDE
"""
Airflow DAG for Fraud Detection Model Training Pipeline
Using Isolation Forest
FIXED VERSION - Better error handling, logging, and MLflow compatibility
"""

from datetime import datetime, timedelta
import os
import sys
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', '/opt/airflow')
CONFIG_PATH = os.path.join(AIRFLOW_HOME, "config/config.yaml")
MODEL_PATH = os.path.join(AIRFLOW_HOME, "src/models/fraud_detection_model.pkl")

# Ensure Python can find your custom modules
sys.path.insert(0, os.path.join(AIRFLOW_HOME, "dags"))
sys.path.insert(0, AIRFLOW_HOME)

logger.info(f"AIRFLOW_HOME: {AIRFLOW_HOME}")
logger.info(f"CONFIG_PATH: {CONFIG_PATH}")
logger.info(f"MODEL_PATH: {MODEL_PATH}")

# Default arguments
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
    description='Train fraud detection Isolation Forest model',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['fraud-detection', 'isolation-forest', 'ml-training'],
)


def validate_environment(**context):
    """Validate that all required files and directories exist"""
    logger.info("="*60)
    logger.info("VALIDATING ENVIRONMENT")
    logger.info("="*60)
    
    validation_errors = []
    
    # Check Python version
    logger.info(f"✅ Python version: {sys.version}")
    logger.info(f"✅ Python executable: {sys.executable}")
    logger.info(f"✅ Working directory: {os.getcwd()}")
    logger.info(f"✅ AIRFLOW_HOME: {AIRFLOW_HOME}")
    
    # Check for config file
    if os.path.exists(CONFIG_PATH):
        logger.info(f"✅ Config found at: {CONFIG_PATH}")
        context['ti'].xcom_push(key='config_path', value=CONFIG_PATH)
    else:
        # Try alternate locations
        alternate_paths = [
            os.path.join(AIRFLOW_HOME, 'src', 'config', 'config.yaml'),
            os.path.join(AIRFLOW_HOME, 'config.yaml'),
            '/opt/airflow/config/config.yaml',
        ]
        
        config_found = False
        for alt_path in alternate_paths:
            logger.info(f"Trying alternate path: {alt_path}")
            if os.path.exists(alt_path):
                logger.info(f"✅ Config found at: {alt_path}")
                context['ti'].xcom_push(key='config_path', value=alt_path)
                config_found = True
                break
        
        if not config_found:
            error_msg = f"❌ Config file not found at {CONFIG_PATH} or alternate locations"
            logger.error(error_msg)
            validation_errors.append(error_msg)
    
    # Check for data directory
    data_paths = [
        os.path.join(AIRFLOW_HOME, 'data'),
        os.path.join(AIRFLOW_HOME, 'dags', 'data'),
    ]
    
    data_found = False
    for data_path in data_paths:
        if os.path.exists(data_path):
            logger.info(f"✅ Data directory found at: {data_path}")
            # Check for transactions_data.csv
            trans_file = os.path.join(data_path, 'transactions_data.csv')
            if os.path.exists(trans_file):
                file_size = os.path.getsize(trans_file) / (1024 * 1024)  # MB
                logger.info(f"✅ transactions_data.csv found ({file_size:.2f} MB)")
                data_found = True
            else:
                logger.warning(f"⚠️  transactions_data.csv not found at {trans_file}")
            break
    
    if not data_found:
        error_msg = f"⚠️  Data directory not found in standard locations"
        logger.warning(error_msg)
        # Don't add to errors - preprocessing will handle this
    
    # Check for models directory (create if doesn't exist)
    models_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(models_dir):
        logger.info(f"Creating models directory: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"✅ Models directory created")
    else:
        logger.info(f"✅ Models directory exists: {models_dir}")
    
    # Check for required Python packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'yaml', 'mlflow'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} available")
        except ImportError:
            error_msg = f"❌ Required package not found: {package}"
            logger.error(error_msg)
            validation_errors.append(error_msg)
    
    # Raise error if critical validations failed
    if validation_errors:
        error_summary = "\n".join(validation_errors)
        raise RuntimeError(f"Environment validation failed:\n{error_summary}")
    
    logger.info("="*60)
    logger.info("✅ ENVIRONMENT VALIDATION COMPLETE")
    logger.info("="*60)
    
    return True


def preprocess_data(**context):
    """Preprocess data with better error handling"""
    logger.info("="*60)
    logger.info("PREPROCESSING DATA")
    logger.info("="*60)
    
    try:
        # Get config path from previous task
        config_path = context['ti'].xcom_pull(key='config_path', task_ids='validate_environment')
        
        if not config_path:
            logger.warning("Config path not in XCom, using default")
            config_path = CONFIG_PATH
        
        logger.info(f"Using config: {config_path}")
        
        # Import preprocessor
        try:
            from ml_training.data_preprocessor import DataPreprocessor
        except ImportError:
            from dags.ml_training.data_preprocessor import DataPreprocessor
        
        # Run preprocessing
        preprocessor = DataPreprocessor(config_path=config_path)
        df = preprocessor.process_all()
        
        # Store metadata in XCom
        context['ti'].xcom_push(key='dataset_shape', value=df.shape)
        context['ti'].xcom_push(key='fraud_rate', value=float(df['is_fraud'].mean()))
        
        logger.info(f"✅ Preprocessing complete! Dataset shape: {df.shape}")
        logger.info(f"✅ Fraud rate: {df['is_fraud'].mean():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        logger.exception("Full traceback:")
        raise


def train_model(**context):
    """Train Isolation Forest model with better error handling"""
    logger.info("="*60)
    logger.info("TRAINING MODEL")
    logger.info("="*60)
    
    try:
        # Get config path
        config_path = context['ti'].xcom_pull(key='config_path', task_ids='validate_environment')
        
        if not config_path:
            logger.warning("Config path not in XCom, using default")
            config_path = CONFIG_PATH
        
        logger.info(f"Using config: {config_path}")
        
        # Import trainer
        try:
            from ml_training.train_model import IsolationForestTrainer
        except ImportError:
            from dags.ml_training.train_model import IsolationForestTrainer
        
        # Run training
        trainer = IsolationForestTrainer(config_path=config_path)
        model, metrics = trainer.run_training_pipeline()
        
        # Store metrics in XCom
        if metrics and '1' in metrics:
            context['ti'].xcom_push(key='precision', value=metrics['1']['precision'])
            context['ti'].xcom_push(key='recall', value=metrics['1']['recall'])
            context['ti'].xcom_push(key='f1_score', value=metrics['1']['f1-score'])
        
        logger.info("="*60)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("="*60)
        
        if metrics:
            logger.info("Training Metrics:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for k, v in value.items():
                        logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
                else:
                    logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.exception("Full traceback:")
        raise


def validate_model(**context):
    """Validate model exists and is loadable"""
    logger.info("="*60)
    logger.info("VALIDATING MODEL")
    logger.info("="*60)
    
    try:
        import pickle
        
        logger.info(f"Looking for model at: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        # Check file size
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        logger.info(f"✅ Model file found ({model_size:.2f} MB)")
        
        # Try to load model to ensure it's not corrupted
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"✅ Model type: {type(model).__name__}")
        logger.info(f"✅ Model loaded successfully")
        
        # Try to check if scaler exists too
        scaler_path = MODEL_PATH.replace('fraud_detection_model.pkl', 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"✅ Scaler found and validated")
        
        logger.info("="*60)
        logger.info("✅ MODEL VALIDATION COMPLETE")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")
        logger.exception("Full traceback:")
        raise


def log_results(**context):
    """Log training results summary"""
    logger.info("="*60)
    logger.info("TRAINING RESULTS SUMMARY")
    logger.info("="*60)
    
    try:
        # Get metrics from XCom
        dataset_shape = context['ti'].xcom_pull(key='dataset_shape', task_ids='preprocess_data')
        fraud_rate = context['ti'].xcom_pull(key='fraud_rate', task_ids='preprocess_data')
        precision = context['ti'].xcom_pull(key='precision', task_ids='train_model')
        recall = context['ti'].xcom_pull(key='recall', task_ids='train_model')
        f1_score = context['ti'].xcom_pull(key='f1_score', task_ids='train_model')
        
        logger.info("Dataset Statistics:")
        if dataset_shape:
            logger.info(f"  Shape: {dataset_shape}")
        if fraud_rate is not None:
            logger.info(f"  Fraud Rate: {fraud_rate:.4f}")
        
        logger.info("\nModel Performance:")
        if precision is not None:
            logger.info(f"  Precision: {precision:.4f}")
        if recall is not None:
            logger.info(f"  Recall: {recall:.4f}")
        if f1_score is not None:
            logger.info(f"  F1-Score: {f1_score:.4f}")
        
        logger.info(f"\nModel Location: {MODEL_PATH}")
        
    except Exception as e:
        logger.warning(f"Could not retrieve all metrics: {e}")
    
    logger.info("="*60)
    logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    
    return True


# Define tasks
validate_task = PythonOperator(
    task_id='validate_environment',
    python_callable=validate_environment,
    provide_context=True,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    provide_context=True,
    dag=dag,
)

log_results_task = PythonOperator(
    task_id='log_results',
    python_callable=log_results,
    provide_context=True,
    dag=dag,
)

notify_task = BashOperator(
    task_id='send_notification',
    bash_command='echo "✅ Isolation Forest model training completed successfully!"',
    dag=dag,
)

# Define task dependencies
validate_task >> preprocess_task >> train_task >> validate_model_task >> log_results_task >> notify_task