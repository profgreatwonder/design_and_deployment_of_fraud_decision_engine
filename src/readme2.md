# Complete Fraud Detection System - File List & Setup

## 📋 All Files to Create

### 1. Configuration
```
src/config/config.yaml
```

### 2. ML Training
```
src/ml_training/data_preprocessor.py
src/ml_training/train_model.py
src/ml_training/requirements.txt
```

### 3. Decisioning Engine
```
src/decisioning_engine/decision_engine.py
src/decisioning_engine/requirements.txt
```

### 4. Producer
```
src/producer/Dockerfile
src/producer/main.py
src/producer/requirements.txt
```

### 5. Consumer
```
src/consumer/Dockerfile
src/consumer/main.py
src/consumer/requirements.txt
```

### 6. Streamlit UI
```
src/streamlit_app/Dockerfile
src/streamlit_app/app.py
src/streamlit_app/requirements.txt
```

### 7. Airflow
```
src/dags/fraud_detection_pipeline.py
```

### 8. Docker Configuration
```
docker-compose.yml
.env
```

### 9. Documentation
```
README.md
```

## 🚀 Setup Instructions

### Prerequisites
Your folder structure should look like this:

```
design_and_deployment_of_fraud_detection_system/
├── data/                     # YOUR EXISTING DATA
│   ├── cards_data.csv
│   ├── mcc_codes.json
│   ├── transactions_data.csv
│   ├── users_data.csv
│   └── train_fraud_labels.json
├── src/                      # CREATE THESE
│   ├── config/
│   ├── ml_training/
│   ├── decisioning_engine/
│   ├── producer/
│   ├── consumer/
│   ├── streamlit_app/
│   ├── dags/
│   └── models/              # Will be created automatically
├── docker-compose.yml        # CREATE THIS
├── .env                      # CREATE THIS
└── README.md                 # CREATE THIS
```

### Step-by-Step Setup

#### 1. Create Directory Structure
```bash
cd design_and_deployment_of_fraud_detection_system

# Create all directories
mkdir -p src/{config,ml_training,decisioning_engine,producer,consumer,streamlit_app,dags,models}
```

#### 2. Create All Python Files

Copy the code I provided into these files:

**src/config/config.yaml**
- Contains all system configuration

**src/ml_training/data_preprocessor.py**
- Loads data from `data/` folder
- Merges all datasets
- Engineers features

**src/ml_training/train_model.py**
- Trains GRU model
- Saves to `src/models/`

**src/decisioning_engine/decision_engine.py**
- Real-time fraud detection
- Returns ALLOW/REJECT/PEND

**src/producer/main.py**
- Reads from `data/` folder
- Sends to Kafka

**src/consumer/main.py**
- Consumes from Kafka
- Applies fraud detection

**src/streamlit_app/app.py**
- Web UI for testing

**src/dags/fraud_detection_pipeline.py**
- Airflow DAG for training

#### 3. Create Requirements Files

**src/ml_training/requirements.txt**
```
tensorflow==2.15.0
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
pyyaml==6.0.1
imbalanced-learn==0.11.0
mlflow==2.9.2
psycopg2-binary==2.9.9
boto3==1.34.34
```

**src/decisioning_engine/requirements.txt**
```
tensorflow==2.15.0
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
pyyaml==6.0.1
```

**src/producer/requirements.txt**
```
confluent-kafka==2.3.0
python-dotenv==1.0.0
pandas==2.0.3
pyyaml==6.0.1
```

**src/consumer/requirements.txt**
```
confluent-kafka==2.3.0
python-dotenv==1.0.0
pyyaml==6.0.1
tensorflow==2.15.0
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
```

**src/streamlit_app/requirements.txt**
```
streamlit==1.29.0
tensorflow==2.15.0
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
pyyaml==6.0.1
plotly==5.18.0
```

#### 4. Create Dockerfiles

**src/producer/Dockerfile**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]
```

**src/consumer/Dockerfile**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]
```

**src/streamlit_app/Dockerfile**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### 5. Create docker-compose.yml and .env

Copy the `docker-compose.yml` and `.env` files I provided.

#### 6. Deploy

```bash
# Build services
docker-compose build

# Start infrastructure
docker-compose up -d postgres redis flower zookeeper kafka kafka-ui mc minio mlflow-server

# Wait 30 seconds for services to start

# Initialize Airflow
docker-compose up airflow-init

# Start Airflow
docker-compose up -d airflow-webserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer airflow-cli

# Access Airflow at http://localhost:8080
# Trigger the training DAG

# After training completes, start real-time components
docker-compose up -d producer consumer streamlit

# Access Streamlit at http://localhost:8501
```

## ✅ Verification Checklist

- [ ] All directories created
- [ ] All Python files created with correct code
- [ ] All requirements.txt files created
- [ ] All Dockerfiles created
- [ ] docker-compose.yml created
- [ ] .env file created
- [ ] Data files exist in `data/` folder
- [ ] Docker and Docker Compose installed
- [ ] Services started successfully
- [ ] Model trained via Airflow
- [ ] Streamlit UI accessible

## 🎯 Key Differences from Your Original Code

### What Changed:
1. **Removed Aiven Kafka** → Using local Kafka
2. **No SSL certificates needed** → Plain authentication
3. **Uses YOUR dataset** → Reads from `data/` folder
4. **No synthetic data** → All transactions from your files
5. **Added decisioning logic** → ALLOW/REJECT/PEND
6. **Added Streamlit UI** → Easy testing interface
7. **Fixed data leakage** → Past-only features
8. **Integrated MLflow** → Experiment tracking

### What Stayed the Same:
- GRU model architecture
- Feature engineering approach
- Data preprocessing logic
- SMOTE for class imbalance
- Airflow orchestration

## 🚨 Important Notes

1. **src is the main folder**: All application code goes in `src/`
2. **data folder contains YOUR dataset**: No changes needed
3. **Local Kafka**: No external services required
4. **Models auto-saved**: Training saves to `src/models/`
5. **Consistent paths**: All imports use relative paths

## 📞 Quick Commands Reference

```bash
# View logs
docker-compose logs -f consumer

# Restart a service
docker-compose restart consumer

# Stop all
docker-compose down

# Rebuild
docker-compose up -d --build

# Access container
docker-compose exec consumer bash

# Check service health
docker-compose ps
```

## 🎓 Understanding the Flow

1. **Training** (Airflow):
   - Loads data from `data/` folder
   - Preprocesses and engineers features
   - Trains GRU model
   - Saves to `src/models/`

2. **Real-time** (Kafka + Consumer):
   - Producer reads from dataset → Kafka
   - Consumer gets message → Decision Engine
   - Decision Engine loads model → Predicts
   - Returns ALLOW/REJECT/PEND

3. **Testing** (Streamlit):
   - Enter transaction details
   - Calls Decision Engine directly
   - Shows probability and decision
   - Tracks history

All components use the SAME model, scaler, and encoder from `src/models/`.