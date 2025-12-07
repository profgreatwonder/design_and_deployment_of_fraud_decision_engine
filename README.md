# Design and Deployment of a Low-Latency, Real-Time Fraud Decisioning Engine


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Project Goal](#project-goal)
- [Solution Approach](#solution-approach)
- [Dataset Description](#dataset-description)
- [Project Architecture](#project-architecture)
- [Technology Stack](#technology-stack)
- [Replicating the project](#replicating-the-project)
- [Installation & Setup](#installation--setup)
- [Deployment](#deployment)
- [Challenges & Solutions](#challenges--solutions)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Problem Statement

Financial institutions lose billions annually to fraudulent transactions. While many fraud detection models exist, the critical business challenge lies in **operationalizing** these models into a reliable, low-latency system that can make accurate decisions (ALLOW, REJECT, PEND) in real-time, balancing fraud prevention with customer experience.

### Key Business Requirements:
- **Low Latency**: Decision within <100ms for real-time transactions
- **High Recall**: Catch 90%+ of fraud cases
- **Balanced Precision**: Minimize false positives to maintain customer experience
- **Scalability**: Handle millions of transactions per day
- **Real-time Updates**: Adapt to emerging fraud patterns

---
## Project Goal

Build a fraud decisioning system that consists of:
1. **ML System**: Trained on financial transaction records to predict fraud probability
2. **Real-Time Decisioning Engine**: Interprets ML predictions and returns decisions (ALLOW, REJECT, PEND) with low latency

The system will be demonstrated through a web frontend that simulates payment transactions and displays the engine's decision in real-time.

---

## Solution Approach

### End-to-End Fraud Intelligence System

#### 1. Machine Learning Track
- Build and train high-recall ML model using isolation forest and LSTM/RNN architecture
- Predict probability of fraudulent transactions
- Handle severe class imbalance (97% legitimate, 3% fraud)
- Optimize for recall while maintaining acceptable precision

#### 2. Production AI Track
- Architect scalable, real-time decisioning engine
- Implement dynamic business rules
- Deploy model with <100ms latency
- Stream processing with Apache Kafka
- Automated retraining pipeline with Apache Airflow

#### 3. Demonstration Layer
- Web frontend for transaction simulation
- Real-time decision display
- API documentation and testing interface

---

## Dataset Description

### Comprehensive Financial Dataset
A unified financial dataset from a bank covering the 2010s, merging transaction logs, customer profiles, and card records.

**Final Merged Dataset:**
- **Rows**: 23,234 transactions
- **Columns**: 50 features
- **Time Period**: 2010s decade
- **Format**: CSV, JSON

### Dataset Components

#### 1. Transaction Data (`transactions_data.csv`)
- Detailed transaction records with amounts and timestamps
- Merchant information and transaction types
- Core dataset for fraud pattern analysis
```
Key Features:
- transaction_id, amount, timestamp
- merchant_id, merchant_city, merchant_state
- use_chip, mcc (merchant category code)
```

#### 2. Card Information (`cards_data.csv`)
- Credit and debit card details
- Card limits, types, and activation dates
- Links to customer accounts via card_id
```
Key Features:
- card_id, client_id, card_brand, card_type
- credit_limit, has_chip, card_on_dark_web
- acct_open_date, year_pin_last_changed
```

#### 3. Merchant Category Codes (`mcc_codes.json`)
- Standard MCC classification codes
- Business type categorization
- Industry-standard codes with descriptions
```json
{
  "5411": "Grocery Stores",
  "5812": "Eating Places/Restaurants",
  "5814": "Fast Food Restaurants"
}
```

#### 4. Fraud Labels (`train_fraud_labels.json`)
- Binary classification labels (0 = legitimate, 1 = fraud)
- Ground truth for supervised learning
- Highly imbalanced distribution

#### 5. User Data (`users_data.csv`)
- Customer demographic information
- Account-related details
- Enables customer segmentation
```
Key Features:
- client_id, age, gender, income
- credit_score, total_debt, num_credit_cards
- latitude, longitude (for location analysis)
```

#### Merged Data
#### 1. Transaction Information (Core)

id: Unique transaction identifier
date: Transaction timestamp
amount: Transaction value in dollars (can be negative for refunds/reversals)
merchant_id: Unique identifier for the merchant
merchant_city, merchant_state, zip: Merchant location details
mcc: Merchant Category Code (industry classification)
is_fraud: Target variable (1 = fraudulent, 0 = legitimate)

#### 2. Card Details

card_id: Unique card identifier
card_brand: Card network (e.g., Visa, Mastercard)
card_type: Type of card (Credit, Debit, Prepaid Debit)
card_number: Card number (likely masked/hashed)
expires: Card expiration date
cvv: Card security code
has_chip: Whether card has EMV chip capability
use_chip: Whether chip was used for this transaction
card_on_dark_web: Flag indicating if card details were found on dark web

#### 3. Account Information

client_id: Unique customer identifier
num_cards_issued: Number of cards issued to the client
credit_limit: Maximum credit available
acct_open_date: Account opening date
year_pin_last_changed: Year the PIN was last updated
errors: Transaction processing errors (if any)

#### 4. Customer Demographics

current_age: Customer's current age
retirement_age: Expected retirement age
birth_year, birth_month: Date of birth components
gender: Customer gender
address: Customer address
latitude, longitude: Geographic coordinates of customer location

#### 5. Financial Profile

per_capita_income: Income per capita in the customer's area
yearly_income: Customer's annual income
total_debt: Total outstanding debt
credit_score: Credit rating (488-850 range)
num_credit_cards: Number of credit cards held

#### 6. Temporal Features (Engineered)

hour: Hour of transaction (0-23)
day_of_week: Day of week (0-6)
month: Month of transaction (1-12)

#### 7. Derived Risk Features (Engineered)

credit_utilization: Ratio of debt to credit limit

rolling_mean_3day: 3-day rolling average of transaction amounts

rolling_std_3day: 3-day rolling standard deviation (spending volatility)

age_at_acct_open: Customer age when account was opened

card_age: Years since account opening

acct_open_year: Year the account was opened

is_high_risk_mcc: Binary flag for high-risk merchant categories

transactions_per_day_past: Count of prior transactions that day

transactions_per_week_past: Count of prior transactions that week

---

## Project Architecture

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES LAYER                        │
│  Cards │ MCC │ Transactions │ Clients │ Labels              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              DATA INGESTION & PROCESSING                     │
│  • Multi-format loader (JSON, CSV, JSONL)                   │
│  • Data cleaning & standardization                          │
│  • Missing value handling                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              DATA INTEGRATION LAYER                          │
│  • Relational joins (LEFT JOIN strategy)                    │
│  • Feature engineering                                       │
│  • Sequence creation for RNN                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              CLASS BALANCING LAYER                           │
│  • SMOTE oversampling (32:1 → 2:1 ratio)                   │
│  • Stratified train-test split                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              MODEL ARCHITECTURE (LSTM)                       │
│  Input (sequence) → Embedding → LSTM(64) → LSTM(32) →      │
│  Dense(16) → Output (sigmoid)                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│         TRAINING & EVALUATION PIPELINE                       │
│  • MLflow experiment tracking                                │
│  • DVC for data versioning                                  │
│  • Early stopping & model checkpointing                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│         REAL-TIME DECISIONING ENGINE                         │
│  • Apache Kafka for data streaming                          │
│  • Apache Spark for inference                               │
│  • Business rules engine (ALLOW/REJECT/PEND)                │
│  • Model serving with <100ms latency                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              DEPLOYMENT & MONITORING                         │
│  • Render deployment                                         │
│  • MongoDB for logging                                       │
│  • Telegram notifications                                    │
│  • GitHub Actions CI/CD                                      │
└─────────────────────────────────────────────────────────────┘
```

### Architectural Design Principles

This fraud detection system uses a **layered architecture** with **Recurrent Neural Networks (RNN/LSTM)** to identify fraudulent transactions by analyzing sequential patterns in transaction data.

**Key Design Patterns:**
- **Repository Pattern**: Abstracts data sources
- **Pipeline Pattern**: Sequential data transformations
- **Strategy Pattern**: Different handling for different data types
- **Factory Pattern**: Creates appropriate parsers based on file type
- **Observer Pattern**: Callbacks for training monitoring

---

## Technology Stack

### Core Technologies

#### Machine Learning & Data Science
- **Python 3.8+**: Primary programming language
- **TensorFlow 2.x / Keras**: Deep learning framework for LSTM models
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Preprocessing, metrics, and classical ML
- **imbalanced-learn**: SMOTE implementation for class balancing

#### Data Processing & Streaming
- **Apache Kafka**: Real-time data streaming 
- **Apache Spark**: Distributed inference processing
- **Apache Airflow**: Scheduled model retraining with DAGs

#### Experiment Tracking & Versioning
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data version control and pipeline tracking
- **DagsHub**: Collaborative MLOps platform

#### Database & Storage
- **MongoDB**: Transaction logging and results storage
- **Cloud Storage**: Model artifacts and datasets

#### Deployment & DevOps
- **Render**: Model deployment platform
- **GitHub Actions**: CI/CD pipeline automation
- **Self-Hosted Runner**: Custom deployment workflows
- **Docker**: Containerization (optional)

#### Frontend & API
- **Streamlit**: Interactive web interface
- **FastAPI**: RESTful API for model serving

#### Development Tool
- **Git & GitHub**: Version control
- **VS Code**: Primary IDE
- **Google Colab**: Cloud-based development and training and exploratory data analysis
- **Jupyter Notebooks**: Exploratory data analysis

---

## Replicating the project

### Prerequisites
```bash
- Python 3.8 or higher
- pip package manager
- Git
- 8GB+ RAM recommended
- (Optional) GPU for faster training
```

### Environment Setup

#### Option 1: Using Conda (Recommended for Mac)
```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-engine.git
cd fraud-detection-engine

# Create conda environment from requirements
conda env create -f requirements-mac.yaml

# Activate environment
conda activate fraud-detection
```

#### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-engine.git
cd fraud-detection-engine

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt (windows)
conda env export --no-builds > requirements-mac.yaml (mac)
pip list --format=freeze > requirements.txt (update dependency list)
```

#### Option 3: Google Colab (No Installation Required)
```python
# In a Colab notebook
!git clone https://github.com/yourusername/fraud-detection-engine.git
%cd fraud-detection-engine
!pip install -r requirements.txt
```

### Configuration

#### Set up MLflow
```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=https://dagshub.com/yourusername/fraud-detection.mlflow

# Set DagsHub credentials
export MLFLOW_TRACKING_USERNAME=yourusername
export MLFLOW_TRACKING_PASSWORD=your_token
```

#### Set up Kafka (if using streaming)
```bash
# Add Aiven Kafka credentials to .env file
KAFKA_BOOTSTRAP_SERVERS=your-kafka-server.aivencloud.com:12345
KAFKA_SECURITY_PROTOCOL=SSL
KAFKA_SSL_CAFILE=/path/to/ca.pem
```
---
**Challenges**
- Data integration challenges: Several attempts to load data failed with parsing errors. This lead to time loss due to trying different methods.
- Could not load label.json data. This is because, the file was corrupted which caused a lot of delay as it hindered project progress. It could also cause loss of data.
- No common columns across dataset for merging. Redesign had to take place for merging to work.
- Unequal row counts: Different datasets had different row count. 

---
**Challenges Solved:**
1. Credit card numbers had inconsistent formatting (2 digits vs 3)
2. JSON file corruption requiring line-by-line parsing
3. Multiple file formats (CSV, JSON, JSONL)
---
- ## Class Imbalance Analysis

| Class | Count | Percentage | Imbalance Ratio |
|-------|-------|------------|-----------------|
| Legitimate | 48,500 | 97% | 32:1 |
| Fraud | 1,500 | 3% | - |
| **Total** | **50,000** | **100%** | - |
---
### Visualization
```
Legitimate ████████████████████████████████████████████████ 97%
Fraud      ██ 3%
```
---
**# Summary of Findings**
#### Dataset Characteristics

Highly imbalanced: Only 0.18% of transactions are fraudulent
Transaction amounts: Average $43.60, with refunds/reversals (minimum -$500)
Credit scores: Range from 488-850, averaging 714

Fraud Patterns

Card types: Prepaid debit cards show highest fraud rate (0.26%), followed by credit cards (0.20%)

High-risk merchants: Certain MCCs show alarmingly high fraud rates:

Water Transportation (4411): 46.34%
Music Stores (5733): 40%
Several other categories: 14-17%

Model Performance
The GRU model achieved near-perfect results after addressing data leakage issues:

Test accuracy: 99.98%
Test recall: 100%
AUC-ROC: 1.00

#### Key improvements included:

Fixing look-ahead bias in transaction frequency features
Applying SMOTE to balance classes (5 fraud cases → 23,228 each class)
Feature engineering: rolling statistics, card age, high-risk MCC flags
One-hot encoding expanded features from 34 to 2,503 columns

#### Important Caveat
Despite excellent metrics, the near-perfect performance warrants further validation on entirely unseen real-world data to confirm the model isn't benefiting from any remaining data leakage.

---

**Conclusion**
This fraud detection analysis successfully identified critical patterns in fraudulent transactions, particularly highlighting severe vulnerabilities in specific merchant categories like water transportation and music stores, where fraud rates exceed 40%. The development of a GRU-based detection model demonstrates the power of deep learning approaches when combined with carefully engineered features such as transaction frequency patterns, rolling statistics, and merchant risk indicators.

The model's near-perfect performance (99.98% accuracy, 100% recall) is encouraging, especially after addressing data leakage concerns through rigorous feature re-engineering. However, these exceptional results also necessitate cautious interpretation. The extreme class imbalance (0.18% fraud rate) required synthetic oversampling via SMOTE, which while effective for training, may not fully represent real-world complexity.

---

**Recommendations**
Before deployment, the model requires validation on completely independent, real-world transaction data to confirm its generalization capability. Additionally, focusing fraud prevention efforts on the identified high-risk merchant categories could provide immediate practical value while the model undergoes further testing. The methodology established here—combining domain-specific feature engineering with advanced neural networks—provides a solid foundation for operational fraud detection systems, pending final validation.

---
**Future updates**
Model Enhancements

Ensemble methods: Combine GRU with XGBoost/Random Forest
Real-time API: Deploy for instant transaction scoring
Explainability: Add SHAP/LIME for interpretable predictions

Feature Additions

Velocity checks: Track spending spikes in 1-hour/24-hour windows
Geolocation anomalies: Flag unusual transaction locations
Dynamic merchant risk: Update MCC risk scores based on trends

Critical Validations

Independent test set: Validate on 6-12 months of unseen data
A/B testing: Pilot with manual review before full deployment
Data leakage audit: Investigate near-perfect scores thoroughly
Feedback loop: Retrain with confirmed fraud cases

Operations

Risk-based alerts: Prioritize high/medium/low risk transactions
Real-time blocking: Auto-decline high-risk transactions
Monitoring dashboard: Track model performance and fraud trends

Priority: Validate model on completely independent data before production deployment.Claude is AI and can make mistakes. Please double-check responses.
