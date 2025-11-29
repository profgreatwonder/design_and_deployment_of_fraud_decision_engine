# Design and Deployment of a Low-Latency, Real-Time Fraud Decisioning Engine
conda env export --no-builds > requirements-mac.yaml
pip list --format=freeze > requirements.txt
pip install -r requirements.txt


Design and Deployment of a Low-Latency, Real-Time Fraud Decisioning Engine


**# Problem Statement**

Financial institutions lose billions annually to fraudulent transactions. While many fraud detection models exist, the critical business challenge lies in *operationalizing* these models into a reliable, low-latency system that can make accurate decisions (ALLOW, REJECT, PEND) in real-time, balancing fraud prevention with customer experience.

---
**# Project Goal**

Build a fraud decisioning system that consists of a ML system that is trained on financial transaction records and a real-time decisioning engine that interpretes the ML predictions and returns a decision on every incoming transaction (ALLOW, REJECT, PEND) with low latency.

---
**# Solution Approach**

Develop an end-to-end fraud intelligence system. This involves:

1.  **Machine Learning Track:** Building, training, and evaluating a high-precision machine learning model to predict the probability that an incoming transaction is fraudulent.
2.  **Production AI Track:** Architecting, deploying, and serving this model within a scalable, real-time decisioning engine. This engine will interpret the model's prediction against dynamic business rules to output a final decision (ALLOW, REJECT, PEND) with low latency.

The solution will be demonstrated through a simple web frontend that simulates a payment transaction and displays the engine's decision in real-time.

---
***Problems to Solve Using the Dataset**


---
**## Project Description**





---
**Dataset Description**

The dataset used for this project is the comprehensive financial dataset which is a unified financial dataset from a bank covering the 2010s that merges transaction logs, customer profiles, and card records.


**Dataset Components**

1. Transaction Data (transactions_data.csv)

Detailed transaction records including amounts, timestamps, and merchant details
Covers transactions throughout the 2010s
Features transaction types, amounts, and merchant information
Perfect for analyzing spending patterns and building fraud detection models

2. Card Information (cards_dat.csv)

Credit and debit card details
Includes card limits, types, and activation dates
Links to customer accounts via card_id
Essential for understanding customer financial profiles

3. Merchant Category Codes (mcc_codes.json)

Standard classification codes for business types
Enables transaction categorization and spending analysis
Industry-standard MCC codes with descriptions

4. Fraud Labels (train_fraud_labels.json)

Binary classification labels for transactions
Indicates fraudulent vs. legitimate transactions
Ideal for training supervised fraud detection models

5. User Data (users_data)

Demographic information about customers
Account-related details
Enables customer segmentation and personalized analysis


Format: CSV, JSON
Time Period: 2010s decade


---
**Project Todos**

Define the problem statement
Suggest on we intend to solve it
Decide on the dataset we need
Decide on What models are we using
Use evaluation metrics to decide on which of the models is the most suitable
Importing necessary libraries and loading dataset
Prepare and clean data
perform Exploratory Data Analysis (EDA)
Select key features (feature importance)
Setup repository
Setup MLFlow for model tracking
Split data to training, validation, and test
Build Model
Select the best model based on desired metrics
Use either Streamlit as frontend to showcase the user interface or turn to the model to an API for frontend to call as an endpoint.

---
**Software and Tools**

Git and GitHub
VSCODE
Github Actions
Self Hosted Runner
MongoDB for database
Render for deployment
DVC (Data Version Control) for data versioning and pipeline tracking
MFlow and DagHub for experiment tracking and model registration
Kafka
---
Architectural design

---
**Final Project Deliverables Summary**

1.  *A Trained ML Model:* A model file that takes transaction data and outputs a fraud score, validated on a test set with strong performance metrics.
2.  *A Real-Time API:* A documented API endpoint that receives transaction JSON and returns a decision JSON.
3.  *A Simple Frontend:* A UI to interact with the system.
4.  *Source Code & Documentation:* Clean, well-documented code for both the ML pipeline and the serving infrastructure, hosted on GitHub.
---
**Summary of Findings**

---
**Conclusion**



---
**Challenges**

Recommendations

Future updates
