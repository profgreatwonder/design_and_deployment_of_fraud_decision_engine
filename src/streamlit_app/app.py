# """
# Streamlit UI for Fraud Detection Testing
# """

# import sys
# import os

# import json

# import pandas as pd
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go

# from datetime import datetime

# from src.decisioning_engine.decision_engine import FraudDecisionEngine

# # Page config
# st.set_page_config(
#     page_title="Fraud Detection System",
#     page_icon="🛡️",
#     layout="wide"
# )

# # Initialize session state
# if 'decision_history' not in st.session_state:
#     st.session_state.decision_history = []

# # Load decision engine
# @st.cache_resource
# def load_engine():
#     return FraudDecisionEngine()

# try:
#     engine = load_engine()
#     st.success("✅ Fraud Detection Engine Loaded Successfully")
# except Exception as e:
#     st.error(f"❌ Failed to load engine: {str(e)}")
#     st.stop()

# # Title
# st.title("🛡️ Real-time Fraud Detection System")
# st.markdown("---")

# # Sidebar
# st.sidebar.header("Configuration")

# # Decision thresholds display
# st.sidebar.subheader("Decision Thresholds")
# st.sidebar.info(f"🔴 REJECT: Probability ≥ {engine.decision_config['reject_threshold']}")
# st.sidebar.warning(f"🟡 PEND: Probability ≥ {engine.decision_config['pend_threshold']}")
# st.sidebar.success(f"🟢 ALLOW: Probability < {engine.decision_config['pend_threshold']}")

# # Main tabs
# tab1, tab2, tab3 = st.tabs(["💳 Test Transaction", "📊 Statistics", "📝 Decision History"])

# # Tab 1: Test Transaction
# with tab1:
#     st.header("Test a Transaction")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Transaction Details")
        
#         transaction_id = st.text_input("Transaction ID", value=f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}")
#         user_id = st.number_input("User ID", min_value=1000, max_value=9999, value=1234)
#         amount = st.number_input("Amount ($)", min_value=0.01, max_value=100000.0, value=100.0, step=0.01)
#         currency = st.text_input("Currency", value="USD")
#         merchant = st.text_input("Merchant", value="Test Merchant")
#         location = st.text_input("Location (State)", value="CA")
        
#     with col2:
#         st.subheader("Card & Account Details")
        
#         mcc = st.number_input("MCC Code", min_value=0, max_value=9999, value=5411)
#         credit_limit = st.number_input("Credit Limit ($)", min_value=0.0, max_value=100000.0, value=5000.0)
#         card_brand = st.selectbox("Card Brand", ["Visa", "Mastercard", "American Express", "Discover"])
#         card_type = st.selectbox("Card Type", ["Credit", "Debit", "Debit (Prepaid)"])
#         use_chip = st.selectbox("Transaction Type", ["Chip Transaction", "Swipe Transaction", "Online Transaction"])
#         has_chip = st.selectbox("Has Chip", ["YES", "NO"])
#         gender = st.selectbox("Gender", ["M", "F"])
#         card_on_dark_web = st.selectbox("Card on Dark Web", ["No", "Yes"])
    
#     # High-risk MCC indicator
#     if mcc in engine.high_risk_mccs:
#         st.warning(f"⚠️ MCC {mcc} is flagged as HIGH RISK")
    
#     # Submit button
#     if st.button("🔍 Analyze Transaction", type="primary"):
#         # Create transaction dict
#         transaction = {
#             'transaction_id': transaction_id,
#             'user_id': user_id,
#             'amount': amount,
#             'currency': currency,
#             'merchant': merchant,
#             'timestamp': datetime.now().isoformat(),
#             'location': location,
#             'mcc': mcc,
#             'credit_limit': credit_limit,
#             'card_brand': card_brand,
#             'card_type': card_type,
#             'use_chip': use_chip,
#             'has_chip': has_chip,
#             'gender': gender,
#             'card_on_dark_web': card_on_dark_web
#         }
        
#         # Process transaction
#         with st.spinner("Analyzing transaction..."):
#             result = engine.process_transaction(transaction)
        
#         # Display result
#         st.markdown("---")
#         st.subheader("Decision Result")
        
#         decision = result['decision']
#         fraud_prob = result['fraud_probability']
#         latency = result['latency_ms']
        
#         # Decision card
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             if decision == "REJECT":
#                 st.error(f"### 🔴 {decision}")
#             elif decision == "PEND":
#                 st.warning(f"### 🟡 {decision}")
#             else:
#                 st.success(f"### 🟢 {decision}")
        
#         with col2:
#             st.metric("Fraud Probability", f"{fraud_prob:.2%}")
        
#         with col3:
#             st.metric("Latency", f"{latency:.2f} ms")
        
#         # Add to history
#         st.session_state.decision_history.append({
#             'timestamp': datetime.now(),
#             'transaction_id': transaction_id,
#             'amount': amount,
#             'decision': decision,
#             'fraud_probability': fraud_prob,
#             'latency_ms': latency
#         })
        
#         # Display transaction details
#         with st.expander("View Transaction Details"):
#             st.json(transaction)

# # Tab 2: Statistics
# with tab2:
#     st.header("Decision Statistics")
    
#     if st.session_state.decision_history:
#         df_history = pd.DataFrame(st.session_state.decision_history)
        
#         # Summary metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Transactions", len(df_history))
        
#         with col2:
#             reject_count = len(df_history[df_history['decision'] == 'REJECT'])
#             st.metric("Rejected", reject_count)
        
#         with col3:
#             pend_count = len(df_history[df_history['decision'] == 'PEND'])
#             st.metric("Pending", pend_count)
        
#         with col4:
#             allow_count = len(df_history[df_history['decision'] == 'ALLOW'])
#             st.metric("Allowed", allow_count)
        
#         st.markdown("---")
        
#         # Charts
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Decision distribution
#             decision_counts = df_history['decision'].value_counts()
#             fig_pie = px.pie(
#                 values=decision_counts.values,
#                 names=decision_counts.index,
#                 title="Decision Distribution",
#                 color=decision_counts.index,
#                 color_discrete_map={
#                     'ALLOW': '#00CC96',
#                     'PEND': '#FFA15A',
#                     'REJECT': '#EF553B'
#                 }
#             )
#             st.plotly_chart(fig_pie, use_container_width=True)
        
#         with col2:
#             # Fraud probability distribution
#             fig_hist = px.histogram(
#                 df_history,
#                 x='fraud_probability',
#                 nbins=20,
#                 title="Fraud Probability Distribution",
#                 labels={'fraud_probability': 'Fraud Probability'}
#             )
#             st.plotly_chart(fig_hist, use_container_width=True)
        
#         # Average latency
#         avg_latency = df_history['latency_ms'].mean()
#         st.info(f"📊 Average Decision Latency: {avg_latency:.2f} ms")
        
#         # Latency over time
#         fig_latency = px.line(
#             df_history,
#             x='timestamp',
#             y='latency_ms',
#             title="Decision Latency Over Time",
#             labels={'timestamp': 'Time', 'latency_ms': 'Latency (ms)'}
#         )
#         st.plotly_chart(fig_latency, use_container_width=True)
        
#     else:
#         st.info("No transactions tested yet. Go to the 'Test Transaction' tab to start.")

# # Tab 3: Decision History
# with tab3:
#     st.header("Decision History")
    
#     if st.session_state.decision_history:
#         df_history = pd.DataFrame(st.session_state.decision_history)
        
#         # Display table
#         st.dataframe(
#             df_history.sort_values('timestamp', ascending=False),
#             use_container_width=True,
#             column_config={
#                 'timestamp': st.column_config.DatetimeColumn('Timestamp'),
#                 'amount': st.column_config.NumberColumn('Amount', format="$%.2f"),
#                 'fraud_probability': st.column_config.ProgressColumn('Fraud Prob', min_value=0, max_value=1),
#                 'latency_ms': st.column_config.NumberColumn('Latency (ms)', format="%.2f")
#             }
#         )
        
#         # Clear history button
#         if st.button("🗑️ Clear History"):
#             st.session_state.decision_history = []
#             st.rerun()
#     else:
#         st.info("No decision history available.")

# # Footer
# st.markdown("---")
# st.markdown("### About")
# st.info("""
# This fraud detection system uses a GRU (Gated Recurrent Unit) neural network to analyze transactions in real-time.
# The system returns one of three decisions:
# - **ALLOW**: Transaction appears legitimate
# - **PEND**: Transaction requires manual review
# - **REJECT**: Transaction is likely fraudulent
# """)



# """
# Streamlit UI for Fraud Detection Testing
# Updated for Isolation Forest model
# """

# import sys
# import os
# import json
# import pandas as pd
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime

# # Import the updated decision engine
# from src.decisioning_engine.decision_engine import DecisionEngine

# # Page config
# st.set_page_config(
#     page_title="Fraud Detection System",
#     page_icon="🛡️",
#     layout="wide"
# )

# # Initialize session state
# if 'decision_history' not in st.session_state:
#     st.session_state.decision_history = []

# # Load decision engine
# @st.cache_resource
# def load_engine():
#     return DecisionEngine(config_path="/app/src/config/config.yaml")

# try:
#     engine = load_engine()
#     st.success("✅ Isolation Forest Engine Loaded Successfully")
# except Exception as e:
#     st.error(f"❌ Failed to load engine: {str(e)}")
#     st.info("💡 Make sure the model has been trained first. Run the Airflow DAG: `fraud_detection_simple_training`")
#     st.stop()

# # Title
# st.title("🛡️ Real-time Fraud Detection System")
# st.markdown("**Powered by Isolation Forest Anomaly Detection**")
# st.markdown("---")

# # Sidebar
# st.sidebar.header("Configuration")
# st.sidebar.info("🔍 **Model Type**: Isolation Forest (Unsupervised)")
# st.sidebar.info("📊 **Features Used**: Numerical features only")

# # Risk level thresholds
# st.sidebar.subheader("Risk Level Thresholds")
# st.sidebar.error("🔴 HIGH RISK: Score > 0.7")
# st.sidebar.warning("🟡 MEDIUM RISK: Score > 0.5")
# st.sidebar.success("🟢 LOW RISK: Score ≤ 0.5")

# # Main tabs
# tab1, tab2, tab3 = st.tabs(["💳 Test Transaction", "📊 Statistics", "📝 Decision History"])

# # Tab 1: Test Transaction
# with tab1:
#     st.header("Test a Transaction")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Transaction Details")
        
#         amount = st.number_input("Amount ($)", min_value=0.01, max_value=100000.0, value=100.0, step=0.01)
#         credit_limit = st.number_input("Credit Limit ($)", min_value=0.0, max_value=100000.0, value=5000.0)
        
#         # Calculate credit utilization
#         credit_utilization = amount / credit_limit if credit_limit > 0 else 0
#         st.metric("Credit Utilization", f"{credit_utilization:.2%}")
        
#     with col2:
#         st.subheader("Transaction Context")
        
#         hour = st.slider("Hour of Day", 0, 23, 14)
#         day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 3)
#         month = st.slider("Month", 1, 12, 6)
#         mcc = st.number_input("MCC Code", min_value=0, max_value=9999, value=5411)
    
#     st.subheader("Historical Features")
    
#     col3, col4 = st.columns(2)
    
#     with col3:
#         rolling_mean_3day = st.number_input("Avg Amount (3-day)", min_value=0.0, value=100.0)
#         rolling_std_3day = st.number_input("Std Dev (3-day)", min_value=0.0, value=50.0)
#         transactions_per_day = st.number_input("Transactions Today", min_value=0, value=2)
#         transactions_per_week = st.number_input("Transactions This Week", min_value=0, value=10)
    
#     with col4:
#         age_at_acct_open = st.number_input("Age at Account Open", min_value=18, max_value=100, value=35)
#         card_age = st.number_input("Card Age (years)", min_value=0.0, max_value=50.0, value=3.5)
#         is_high_risk_mcc = st.checkbox("High Risk MCC", value=False)
    
#     # Submit button
#     if st.button("🔍 Analyze Transaction", type="primary"):
#         # Create transaction dict with all numerical features
#         transaction_data = {
#             'amount': amount,
#             'credit_limit': credit_limit,
#             'credit_utilization': credit_utilization,
#             'rolling_mean_3day': rolling_mean_3day,
#             'rolling_std_3day': rolling_std_3day,
#             'transactions_per_day_past': transactions_per_day,
#             'transactions_per_week_past': transactions_per_week,
#             'age_at_acct_open': age_at_acct_open,
#             'card_age': card_age,
#             'hour': hour,
#             'day_of_week': day_of_week,
#             'month': month,
#             'is_high_risk_mcc': 1 if is_high_risk_mcc else 0
#         }
        
#         # Process transaction
#         with st.spinner("Analyzing transaction..."):
#             try:
#                 result = engine.predict(transaction_data)
#             except Exception as e:
#                 st.error(f"❌ Prediction failed: {str(e)}")
#                 st.stop()
        
#         # Display result
#         st.markdown("---")
#         st.subheader("Anomaly Detection Result")
        
#         is_fraud = result['is_fraud']
#         fraud_score = result['fraud_score']
#         risk_level = result['risk_level']
#         raw_score = result.get('raw_anomaly_score', 0)
        
#         # Decision card
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             if is_fraud:
#                 st.error("### 🚨 ANOMALY DETECTED")
#             else:
#                 st.success("### ✅ NORMAL TRANSACTION")
        
#         with col2:
#             st.metric("Fraud Score", f"{fraud_score:.2%}")
        
#         with col3:
#             if risk_level == "HIGH":
#                 st.error(f"### 🔴 {risk_level}")
#             elif risk_level == "MEDIUM":
#                 st.warning(f"### 🟡 {risk_level}")
#             else:
#                 st.success(f"### 🟢 {risk_level}")
        
#         # Additional info
#         st.info(f"📊 Raw Anomaly Score: {raw_score:.4f} (lower = more anomalous)")
        
#         # Add to history
#         st.session_state.decision_history.append({
#             'timestamp': datetime.now(),
#             'amount': amount,
#             'is_fraud': is_fraud,
#             'fraud_score': fraud_score,
#             'risk_level': risk_level
#         })
        
#         # Display transaction details
#         with st.expander("View Transaction Features"):
#             st.json(transaction_data)

# # Tab 2: Statistics
# with tab2:
#     st.header("Detection Statistics")
    
#     if st.session_state.decision_history:
#         df_history = pd.DataFrame(st.session_state.decision_history)
        
#         # Summary metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Transactions", len(df_history))
        
#         with col2:
#             fraud_count = df_history['is_fraud'].sum()
#             st.metric("Anomalies Detected", fraud_count)
        
#         with col3:
#             normal_count = (~df_history['is_fraud']).sum()
#             st.metric("Normal Transactions", normal_count)
        
#         with col4:
#             avg_score = df_history['fraud_score'].mean()
#             st.metric("Avg Fraud Score", f"{avg_score:.2%}")
        
#         st.markdown("---")
        
#         # Charts
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Fraud detection distribution
#             fraud_counts = df_history['is_fraud'].value_counts()
#             fraud_counts.index = ['Normal', 'Fraud']
#             fig_pie = px.pie(
#                 values=fraud_counts.values,
#                 names=fraud_counts.index,
#                 title="Detection Distribution",
#                 color=fraud_counts.index,
#                 color_discrete_map={
#                     'Normal': '#00CC96',
#                     'Fraud': '#EF553B'
#                 }
#             )
#             st.plotly_chart(fig_pie, use_container_width=True)
        
#         with col2:
#             # Risk level distribution
#             risk_counts = df_history['risk_level'].value_counts()
#             fig_risk = px.bar(
#                 x=risk_counts.index,
#                 y=risk_counts.values,
#                 title="Risk Level Distribution",
#                 labels={'x': 'Risk Level', 'y': 'Count'},
#                 color=risk_counts.index,
#                 color_discrete_map={
#                     'LOW': '#00CC96',
#                     'MEDIUM': '#FFA15A',
#                     'HIGH': '#EF553B'
#                 }
#             )
#             st.plotly_chart(fig_risk, use_container_width=True)
        
#         # Fraud probability distribution
#         fig_hist = px.histogram(
#             df_history,
#             x='fraud_score',
#             nbins=20,
#             title="Fraud Score Distribution",
#             labels={'fraud_score': 'Fraud Score'}
#         )
#         st.plotly_chart(fig_hist, use_container_width=True)
        
#         # Score over time
#         fig_timeline = px.scatter(
#             df_history,
#             x='timestamp',
#             y='fraud_score',
#             color='risk_level',
#             title="Fraud Score Over Time",
#             labels={'timestamp': 'Time', 'fraud_score': 'Fraud Score'},
#             color_discrete_map={
#                 'LOW': '#00CC96',
#                 'MEDIUM': '#FFA15A',
#                 'HIGH': '#EF553B'
#             }
#         )
#         st.plotly_chart(fig_timeline, use_container_width=True)
        
#     else:
#         st.info("No transactions tested yet. Go to the 'Test Transaction' tab to start.")

# # Tab 3: Decision History
# with tab3:
#     st.header("Detection History")
    
#     if st.session_state.decision_history:
#         df_history = pd.DataFrame(st.session_state.decision_history)
        
#         # Display table
#         st.dataframe(
#             df_history.sort_values('timestamp', ascending=False),
#             use_container_width=True,
#             column_config={
#                 'timestamp': st.column_config.DatetimeColumn('Timestamp'),
#                 'amount': st.column_config.NumberColumn('Amount', format="$%.2f"),
#                 'is_fraud': st.column_config.CheckboxColumn('Anomaly?'),
#                 'fraud_score': st.column_config.ProgressColumn('Fraud Score', min_value=0, max_value=1),
#                 'risk_level': st.column_config.TextColumn('Risk Level')
#             }
#         )
        
#         # Download button
#         csv = df_history.to_csv(index=False)
#         st.download_button(
#             label="📥 Download History (CSV)",
#             data=csv,
#             file_name=f"fraud_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             mime="text/csv"
#         )
        
#         # Clear history button
#         if st.button("🗑️ Clear History"):
#             st.session_state.decision_history = []
#             st.rerun()
#     else:
#         st.info("No detection history available.")

# # Footer
# st.markdown("---")
# st.markdown("### About")
# st.info("""
# This fraud detection system uses **Isolation Forest**, an unsupervised machine learning algorithm for anomaly detection.

# **How it works:**
# - The model learns patterns from normal transactions during training
# - Anomalous transactions that deviate from these patterns are flagged as potential fraud
# - The system returns a fraud score (0-1) and risk level (LOW/MEDIUM/HIGH)

# **Key Features:**
# - ✅ No labeled data required for training
# - ✅ Fast training and prediction
# - ✅ Works well with imbalanced data
# - ✅ Detects novel fraud patterns
# """)




# """
# Streamlit UI for Fraud Detection Testing
# Updated for Isolation Forest model & FraudDecisionEngine
# """

# import sys
# import os
# import json
# import pandas as pd
# import streamlit as st
# import plotly.express as px
# from datetime import datetime

# # --- CRITICAL FIX: Add project root to path for imports ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # --- CRITICAL FIX: Import correct class name (FraudDecisionEngine) ---
# from src.decisioning_engine.decision_engine import FraudDecisionEngine

# # Page config
# st.set_page_config(
#     page_title="Fraud Detection System",
#     page_icon="🛡️",
#     layout="wide"
# )

# # Initialize session state
# if 'decision_history' not in st.session_state:
#     st.session_state.decision_history = []

# # Load decision engine
# @st.cache_resource
# def load_engine():
#     # Use the absolute path to config to be safe
#     config_path = os.path.join(project_root, "src/config/config.yaml")
#     return FraudDecisionEngine(config_path=config_path)

# try:
#     engine = load_engine()
#     st.success("✅ Isolation Forest Engine Loaded Successfully")
# except Exception as e:
#     st.error(f"❌ Failed to load engine: {str(e)}")
#     st.info("💡 Make sure the model has been trained first. Run the Airflow DAG: `fraud_detection_training_pipeline`")
#     st.stop()

# # Title
# st.title("🛡️ Real-time Fraud Detection System")
# st.markdown("**Powered by Isolation Forest Anomaly Detection**")
# st.markdown("---")

# # Sidebar
# st.sidebar.header("Configuration")
# st.sidebar.info("🔍 **Model Type**: Isolation Forest (Unsupervised)")
# st.sidebar.info("📊 **Features Used**: Numerical features only")

# # Risk level thresholds
# st.sidebar.subheader("Risk Level Thresholds")
# st.sidebar.error("🔴 HIGH RISK: Score > 0.7")
# st.sidebar.warning("🟡 MEDIUM RISK: Score > 0.5")
# st.sidebar.success("🟢 LOW RISK: Score ≤ 0.5")

# # Main tabs
# tab1, tab2, tab3 = st.tabs(["💳 Test Transaction", "📊 Statistics", "📝 Decision History"])

# # Tab 1: Test Transaction
# with tab1:
#     st.header("Test a Transaction")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Transaction Details")
        
#         amount = st.number_input("Amount ($)", min_value=0.01, max_value=100000.0, value=100.0, step=0.01)
#         credit_limit = st.number_input("Credit Limit ($)", min_value=0.0, max_value=100000.0, value=5000.0)
        
#         # Calculate credit utilization
#         credit_utilization = amount / credit_limit if credit_limit > 0 else 0
#         st.metric("Credit Utilization", f"{credit_utilization:.2%}")
        
#     with col2:
#         st.subheader("Transaction Context")
        
#         hour = st.slider("Hour of Day", 0, 23, 14)
#         day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 3)
#         month = st.slider("Month", 1, 12, 6)
#         mcc = st.number_input("MCC Code", min_value=0, max_value=9999, value=5411)
    
#     st.subheader("Historical Features")
    
#     col3, col4 = st.columns(2)
    
#     with col3:
#         rolling_mean_3day = st.number_input("Avg Amount (3-day)", min_value=0.0, value=100.0)
#         rolling_std_3day = st.number_input("Std Dev (3-day)", min_value=0.0, value=50.0)
#         transactions_per_day = st.number_input("Transactions Today", min_value=0, value=2)
#         transactions_per_week = st.number_input("Transactions This Week", min_value=0, value=10)
    
#     with col4:
#         age_at_acct_open = st.number_input("Age at Account Open", min_value=18, max_value=100, value=35)
#         card_age = st.number_input("Card Age (years)", min_value=0.0, max_value=50.0, value=3.5)
#         is_high_risk_mcc = st.checkbox("High Risk MCC", value=False)
    
#     # Submit button
#     if st.button("🔍 Analyze Transaction", type="primary"):
#         # Create transaction dict with all numerical features
#         transaction_data = {
#             'amount': amount,
#             'credit_limit': credit_limit,
#             'credit_utilization': credit_utilization,
#             'rolling_mean_3day': rolling_mean_3day,
#             'rolling_std_3day': rolling_std_3day,
#             'transactions_per_day_past': transactions_per_day,
#             'transactions_per_week_past': transactions_per_week,
#             'age_at_acct_open': age_at_acct_open,
#             'card_age': card_age,
#             'hour': hour,
#             'day_of_week': day_of_week,
#             'month': month,
#             'is_high_risk_mcc': 1 if is_high_risk_mcc else 0
#         }
        
#         # Process transaction
#         with st.spinner("Analyzing transaction..."):
#             try:
#                 result = engine.predict(transaction_data)
#             except Exception as e:
#                 st.error(f"❌ Prediction failed: {str(e)}")
#                 st.stop()
        
#         # Display result
#         st.markdown("---")
#         st.subheader("Anomaly Detection Result")
        
#         is_fraud = result['is_fraud']
#         fraud_score = result['fraud_score']
        
#         # --- FIX: Calculate Risk Level manually (engine doesn't return it) ---
#         if fraud_score > 0.7:
#             risk_level = "HIGH"
#         elif fraud_score > 0.5:
#             risk_level = "MEDIUM"
#         else:
#             risk_level = "LOW"
            
#         raw_score = result.get('raw_anomaly_score', 0)
        
#         # Decision card
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             if is_fraud:
#                 st.error("### 🚨 ANOMALY DETECTED")
#             else:
#                 st.success("### ✅ NORMAL TRANSACTION")
        
#         with col2:
#             st.metric("Fraud Score", f"{fraud_score:.2%}")
        
#         with col3:
#             if risk_level == "HIGH":
#                 st.error(f"### 🔴 {risk_level}")
#             elif risk_level == "MEDIUM":
#                 st.warning(f"### 🟡 {risk_level}")
#             else:
#                 st.success(f"### 🟢 {risk_level}")
        
#         # Additional info
#         st.info(f"📊 Raw Anomaly Score: {raw_score:.4f} (lower = more anomalous)")
        
#         # Add to history
#         st.session_state.decision_history.append({
#             'timestamp': datetime.now(),
#             'amount': amount,
#             'is_fraud': is_fraud,
#             'fraud_score': fraud_score,
#             'risk_level': risk_level
#         })
        
#         # Display transaction details
#         with st.expander("View Transaction Features"):
#             st.json(transaction_data)

# # Tab 2: Statistics
# with tab2:
#     st.header("Detection Statistics")
    
#     if st.session_state.decision_history:
#         df_history = pd.DataFrame(st.session_state.decision_history)
        
#         # Summary metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Transactions", len(df_history))
        
#         with col2:
#             fraud_count = df_history['is_fraud'].sum()
#             st.metric("Anomalies Detected", fraud_count)
        
#         with col3:
#             normal_count = (~df_history['is_fraud']).sum()
#             st.metric("Normal Transactions", normal_count)
        
#         with col4:
#             avg_score = df_history['fraud_score'].mean()
#             st.metric("Avg Fraud Score", f"{avg_score:.2%}")
        
#         st.markdown("---")
        
#         # Charts
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Fraud detection distribution
#             fraud_counts = df_history['is_fraud'].value_counts()
#             # Map boolean to string for cleaner charts
#             fraud_counts.index = fraud_counts.index.map({True: 'Fraud', False: 'Normal'})
            
#             fig_pie = px.pie(
#                 values=fraud_counts.values,
#                 names=fraud_counts.index,
#                 title="Detection Distribution",
#                 color=fraud_counts.index,
#                 color_discrete_map={
#                     'Normal': '#00CC96',
#                     'Fraud': '#EF553B'
#                 }
#             )
#             st.plotly_chart(fig_pie, use_container_width=True)
        
#         with col2:
#             # Risk level distribution
#             if 'risk_level' in df_history.columns:
#                 risk_counts = df_history['risk_level'].value_counts()
#                 fig_risk = px.bar(
#                     x=risk_counts.index,
#                     y=risk_counts.values,
#                     title="Risk Level Distribution",
#                     labels={'x': 'Risk Level', 'y': 'Count'},
#                     color=risk_counts.index,
#                     color_discrete_map={
#                         'LOW': '#00CC96',
#                         'MEDIUM': '#FFA15A',
#                         'HIGH': '#EF553B'
#                     }
#                 )
#                 st.plotly_chart(fig_risk, use_container_width=True)
        
#         # Fraud probability distribution
#         fig_hist = px.histogram(
#             df_history,
#             x='fraud_score',
#             nbins=20,
#             title="Fraud Score Distribution",
#             labels={'fraud_score': 'Fraud Score'}
#         )
#         st.plotly_chart(fig_hist, use_container_width=True)
        
#     else:
#         st.info("No transactions tested yet. Go to the 'Test Transaction' tab to start.")

# # Tab 3: Decision History
# with tab3:
#     st.header("Detection History")
    
#     if st.session_state.decision_history:
#         df_history = pd.DataFrame(st.session_state.decision_history)
        
#         # Display table
#         st.dataframe(
#             df_history.sort_values('timestamp', ascending=False),
#             use_container_width=True,
#             column_config={
#                 'timestamp': st.column_config.DatetimeColumn('Timestamp'),
#                 'amount': st.column_config.NumberColumn('Amount', format="$%.2f"),
#                 'is_fraud': st.column_config.CheckboxColumn('Anomaly?'),
#                 'fraud_score': st.column_config.ProgressColumn('Fraud Score', min_value=0, max_value=1),
#                 'risk_level': st.column_config.TextColumn('Risk Level')
#             }
#         )
        
#         # Download button
#         csv = df_history.to_csv(index=False)
#         st.download_button(
#             label="📥 Download History (CSV)",
#             data=csv,
#             file_name=f"fraud_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             mime="text/csv"
#         )
        
#         # Clear history button
#         if st.button("🗑️ Clear History"):
#             st.session_state.decision_history = []
#             st.rerun()
#     else:
#         st.info("No detection history available.")

# # Footer
# st.markdown("---")







"""
Streamlit UI for Fraud Detection Testing
Updated for Isolation Forest model & FraudDecisionEngine
Shows Decision: ALLOW / PEND / REJECT
"""

import sys
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
import uuid

# --- CRITICAL FIX: Add project root to path for imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import correct class name
from src.decisioning_engine.decision_engine import FraudDecisionEngine

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'decision_history' not in st.session_state:
    st.session_state.decision_history = []

# Load decision engine
@st.cache_resource
def load_engine():
    # Use the absolute path to config to be safe
    config_path = os.path.join(project_root, "src/config/config.yaml")
    return FraudDecisionEngine(config_path=config_path)

try:
    engine = load_engine()
    st.success("✅ Isolation Forest Engine Loaded Successfully")
except Exception as e:
    st.error(f"❌ Failed to load engine: {str(e)}")
    st.info("💡 Make sure the model has been trained first. Run the Airflow DAG: `fraud_detection_training_pipeline`")
    st.stop()

# Title
st.title("🛡️ Real-time Fraud Decisioning")
st.markdown("**Powered by Isolation Forest Anomaly Detection**")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.info("🔍 **Model Type**: Isolation Forest")
st.sidebar.info("📊 **Features Used**: Numerical features only")

# Decision Thresholds Display
st.sidebar.subheader("Decision Logic")
st.sidebar.error("⛔ REJECT: Score ≥ 0.75")
st.sidebar.warning("⏸️ PEND: Score ≥ 0.45")
st.sidebar.success("✅ ALLOW: Score < 0.45")

# Main tabs
tab1, tab2, tab3 = st.tabs(["💳 Test Transaction", "📊 Statistics", "📝 Decision History"])

# Tab 1: Test Transaction
with tab1:
    st.header("Test a Transaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        
        amount = st.number_input("Amount ($)", min_value=0.01, max_value=100000.0, value=100.0, step=0.01)
        credit_limit = st.number_input("Credit Limit ($)", min_value=0.0, max_value=100000.0, value=5000.0)
        
        # Calculate credit utilization
        credit_utilization = amount / credit_limit if credit_limit > 0 else 0
        st.metric("Credit Utilization", f"{credit_utilization:.2%}")
        
    with col2:
        st.subheader("Transaction Context")
        
        hour = st.slider("Hour of Day", 0, 23, 14)
        day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 3)
        month = st.slider("Month", 1, 12, 6)
        mcc = st.number_input("MCC Code", min_value=0, max_value=9999, value=5411)
    
    st.subheader("Historical Features")
    
    col3, col4 = st.columns(2)
    
    with col3:
        rolling_mean_3day = st.number_input("Avg Amount (3-day)", min_value=0.0, value=100.0)
        rolling_std_3day = st.number_input("Std Dev (3-day)", min_value=0.0, value=50.0)
        transactions_per_day = st.number_input("Transactions Today", min_value=0, value=2)
        transactions_per_week = st.number_input("Transactions This Week", min_value=0, value=10)
    
    with col4:
        age_at_acct_open = st.number_input("Age at Account Open", min_value=18, max_value=100, value=35)
        card_age = st.number_input("Card Age (years)", min_value=0.0, max_value=50.0, value=3.5)
        is_high_risk_mcc = st.checkbox("High Risk MCC", value=False)
    
    # Submit button
    if st.button("🔍 Make Decision", type="primary"):
        # Generate a dummy ID for tracking
        txn_id = f"test_{uuid.uuid4().hex[:8]}"
        
        # Create transaction dict with all numerical features
        transaction_data = {
            'transaction_id': txn_id,
            'amount': amount,
            'credit_limit': credit_limit,
            'credit_utilization': credit_utilization,
            'rolling_mean_3day': rolling_mean_3day,
            'rolling_std_3day': rolling_std_3day,
            'transactions_per_day_past': transactions_per_day,
            'transactions_per_week_past': transactions_per_week,
            'age_at_acct_open': age_at_acct_open,
            'card_age': card_age,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_high_risk_mcc': 1 if is_high_risk_mcc else 0
        }
        
        # Process transaction
        with st.spinner("Processing decision..."):
            try:
                # --- KEY CHANGE: Use process_transaction instead of predict ---
                # This ensures we get the ALLOW/PEND/REJECT decision logic
                result = engine.process_transaction(transaction_data)
            except Exception as e:
                st.error(f"❌ Decision Engine Error: {str(e)}")
                st.stop()
        
        # Display result
        st.markdown("---")
        st.subheader("Decision Engine Result")
        
        # Extract fields from result
        decision = result['decision']          # ALLOW / PEND / REJECT
        fraud_prob = result['fraud_probability']
        raw_score = result.get('raw_score', 0)
        latency = result.get('latency_ms', 0)
        
        # Create visual cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Decision")
            if decision == "REJECT":
                st.error(f"# ⛔ {decision}")
            elif decision == "PEND":
                st.warning(f"# ⏸️ {decision}")
            else:
                st.success(f"# ✅ {decision}")
        
        with col2:
            st.metric("Fraud Probability", f"{fraud_prob:.2%}")
            st.caption("Thresholds: Reject≥75%, Pend≥45%")
        
        with col3:
            st.metric("Latency", f"{latency:.2f} ms")
        
        # Additional info
        st.info(f"📊 Raw Anomaly Score: {raw_score:.4f} (lower = more anomalous)")
        
        # Add to history
        st.session_state.decision_history.append({
            'timestamp': datetime.now(),
            'transaction_id': txn_id,
            'amount': amount,
            'decision': decision,
            'fraud_prob': fraud_prob,
            'latency_ms': latency
        })
        
        # Display transaction details
        with st.expander("View Transaction Payload"):
            st.json(transaction_data)

# Tab 2: Statistics
with tab2:
    st.header("Decision Statistics")
    
    if st.session_state.decision_history:
        df_history = pd.DataFrame(st.session_state.decision_history)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Decisions", len(df_history))
        
        with col2:
            reject_count = (df_history['decision'] == 'REJECT').sum()
            st.metric("Rejected", reject_count)
        
        with col3:
            pend_count = (df_history['decision'] == 'PEND').sum()
            st.metric("Pending Review", pend_count)
            
        with col4:
            allow_count = (df_history['decision'] == 'ALLOW').sum()
            st.metric("Allowed", allow_count)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Decision distribution
            decision_counts = df_history['decision'].value_counts()
            
            fig_pie = px.pie(
                values=decision_counts.values,
                names=decision_counts.index,
                title="Decision Distribution",
                color=decision_counts.index,
                color_discrete_map={
                    'ALLOW': '#00CC96',
                    'PEND': '#FFA15A',
                    'REJECT': '#EF553B'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Latency over time
            fig_latency = px.line(
                df_history,
                x='timestamp',
                y='latency_ms',
                title="Engine Latency (ms)",
                markers=True
            )
            st.plotly_chart(fig_latency, use_container_width=True)
            
    else:
        st.info("No decisions made yet. Go to the 'Test Transaction' tab to start.")

# Tab 3: Decision History
with tab3:
    st.header("Full History")
    
    if st.session_state.decision_history:
        df_history = pd.DataFrame(st.session_state.decision_history)
        
        # Display table
        st.dataframe(
            df_history.sort_values('timestamp', ascending=False),
            use_container_width=True,
            column_config={
                'timestamp': st.column_config.DatetimeColumn('Timestamp', format="D MMM, HH:mm:ss"),
                'transaction_id': st.column_config.TextColumn('Txn ID'),
                'amount': st.column_config.NumberColumn('Amount', format="$%.2f"),
                'decision': st.column_config.TextColumn('Decision'),
                'fraud_prob': st.column_config.ProgressColumn('Fraud Prob', min_value=0, max_value=1, format="%.2f"),
                'latency_ms': st.column_config.NumberColumn('Latency (ms)', format="%.2f")
            }
        )
        
        # Download button
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="📥 Download History (CSV)",
            data=csv,
            file_name=f"decision_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.decision_history = []
            st.rerun()
    else:
        st.info("No history available.")