# Fraud Detection System - Complete Project Structure

## Project Overview
A real-time fraud detection system with ML training pipeline and decisioning engine that returns ALLOW, REJECT, or PEND decisions.

## Folder Structure
```
design_and_deployment_of_fraud_detection_system/
├── data/                          # Your dataset folder
│   ├── cards_data.csv
│   ├── mcc_codes.json
│   ├── transactions_data.csv
│   ├── users_data.csv
│   └── train_fraud_labels.json
├── src/
│   ├── config/
│   │   └── config.yaml
│   ├── consumer/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── producer/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── ml_training/
│   │   ├── train_model.py
│   │   ├── data_preprocessor.py
│   │   └── requirements.txt
│   ├── decisioning_engine/
│   │   ├── decision_engine.py
│   │   ├── model_loader.py
│   │   └── requirements.txt
│   ├── streamlit_app/
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── dags/
│   │   └── fraud_detection_pipeline.py
│   └── models/                    # Trained models stored here
├── docker-compose.yml
├── .env
└── README.md
```

<!-- ## Key Components

### 1. Data Processing (src/dags/ml_training/data_preprocessor.py)
- Reads from `data/` folder (your existing dataset)
- Preprocesses and merges all data files
- Creates engineered features without data leakage

### 2. Model Training (src/dags/ml_training/train_model.py)
- Trains GRU model on historical data
- SMOTE for imbalance
- Stores model artifacts in `src/models/` for MLflow tracking
- Orchestrated by Airflow

### 3. Real-time Transaction Processing - Kafka Producer (src/producer/main.py) and Kafka Consumer (src/consumer/main.py)
- Local Kafka for message streaming
- Producer sends transactions
- Consumer processes with decisioning engine

### 4. Decisioning Engine (src/decisioning_engine/decision_engine.py)
- Loads trained model
- Returns: ALLOW, REJECT, or PEND
- Low-latency predictions (<100ms)

### 5. Streamlit UI
- Test individual transactions
- View model performance
- Monitor real-time decisions
- View statistics
- Decision history

## Airflow DAG (src/dags/fraud_detection_pipeline.py)
Automated training pipeline
Weekly retraining

## Docker Setup
Local Kafka cluster
All services containerized
Easy deployment -->


Here is the updated summary reflecting the successful implementation of the **Isolation Forest** pipeline:

## Key Components

### 1. Data Processing (`src/dags/ml_training/data_preprocessor.py`)
* **Optimized Loading:** Reads from the `data/` folder, utilizing memory-efficient types and row limits (1 million rows) to ensure stability within Docker resource limits.
* **Preprocessing:** cleans raw data by handling missing values and converting currency strings to numeric types.
* **Feature Engineering:** Generates critical numerical features for anomaly detection, including `credit_utilization`, `rolling_mean_3day`, and `transactions_per_day_past`.

### 2. Model Training (`src/dags/ml_training/train_model.py`)
* **Algorithm:** Trains an **Isolation Forest** model (unsupervised learning) to detect anomalies, replacing the previous GRU deep learning approach.
* **Artifact Management:** Automatically saves the trained model (`fraud_detection_model.pkl`) and scaler (`scaler.pkl`) to shared storage for the Consumer.
* **Tracking:** Registers models and parameters (e.g., contamination=0.01) to **MLflow** for version control and lineage.

### 3. Real-time Transaction Processing
* **Kafka Producer (`src/producer/main.py`):** Streams transaction data into the local Kafka cluster.
* **Kafka Consumer (`src/consumer/main.py`):** Consumes live messages and utilizes the `FraudDecisionEngine` to evaluate transactions in real-time.

### 4. Decisioning Engine (`src/decisioning_engine/decision_engine.py`)
* **Inference:** Loads the Isolation Forest artifacts to calculate anomaly scores.
* **Logic:** Maps anomaly scores to decisions based on config thresholds:
    * **REJECT:** Score ≥ 0.70
    * **PEND:** Score ≥ 0.50
    * **ALLOW:** Score < 0.50.

### 5. Streamlit UI (`src/streamlit_app/app.py`)
* **Interactive Testing:** Provides a web interface to manually input transaction details and see immediate model predictions.
* **Visualization:** Displays real-time statistics, fraud score distributions, and risk level breakdowns (High/Medium/Low).

## Airflow DAG (`src/dags/fraud_detection_pipeline.py`)
* **Pipeline:** Automated workflow that handles Preprocessing, Training, Validation, and Notification.
* **Schedule:** Runs daily (`@daily`) to update the model baseline.

## Docker Setup (`docker-compose.yaml`)
* **Infrastructure:** Containerizes all services (Airflow, Kafka, MLflow, Postgres, Redis, Minio) for a unified, portable deployment.
* **Networking:** Configured with internal service discovery (`mlflow-server`, `kafka`) to ensure seamless communication between the training pipeline and operational components.



## Files to Create

### Core Application Files
1. `src/config/config.yaml` - Configuration settings
2. `src/ml_training/train_model.py` - Model training script
3. `src/ml_training/data_preprocessor.py` - Data loading and preprocessing
4. `src/decisioning_engine/decision_engine.py` - Decision logic
5. `src/decisioning_engine/model_loader.py` - Model loading utility
6. `src/producer/main.py` - Kafka producer (local)
7. `src/consumer/main.py` - Kafka consumer with decisioning
8. `src/streamlit_app/app.py` - Streamlit UI
9. `src/dags/fraud_detection_pipeline.py` - Airflow DAG
10. `docker-compose.yml` - Updated for local Kafka
11. `.env` - Environment variables



Complete System Components:

Configuration (src/config/config.yaml)

All system settings in one place
Decision thresholds, features, high-risk MCCs


## Streamlit UI (src/streamlit_app/app.py)
Test individual transactions
View statistics
Decision history


## Airflow DAG (src/dags/fraud_detection_pipeline.py)
Automated training pipeline
Weekly retraining


## Docker Setup
Local Kafka cluster
All services containerized
Easy deployment


conda install -c conda-forge airflow tensorflow streamlit plotly

