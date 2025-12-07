"""
Streamlit UI for Fraud Detection Testing
"""

import sys
import os

import json

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

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
    return FraudDecisionEngine()

try:
    engine = load_engine()
    st.success("✅ Fraud Detection Engine Loaded Successfully")
except Exception as e:
    st.error(f"❌ Failed to load engine: {str(e)}")
    st.stop()

# Title
st.title("🛡️ Real-time Fraud Detection System")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Decision thresholds display
st.sidebar.subheader("Decision Thresholds")
st.sidebar.info(f"🔴 REJECT: Probability ≥ {engine.decision_config['reject_threshold']}")
st.sidebar.warning(f"🟡 PEND: Probability ≥ {engine.decision_config['pend_threshold']}")
st.sidebar.success(f"🟢 ALLOW: Probability < {engine.decision_config['pend_threshold']}")

# Main tabs
tab1, tab2, tab3 = st.tabs(["💳 Test Transaction", "📊 Statistics", "📝 Decision History"])

# Tab 1: Test Transaction
with tab1:
    st.header("Test a Transaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        
        transaction_id = st.text_input("Transaction ID", value=f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        user_id = st.number_input("User ID", min_value=1000, max_value=9999, value=1234)
        amount = st.number_input("Amount ($)", min_value=0.01, max_value=100000.0, value=100.0, step=0.01)
        currency = st.text_input("Currency", value="USD")
        merchant = st.text_input("Merchant", value="Test Merchant")
        location = st.text_input("Location (State)", value="CA")
        
    with col2:
        st.subheader("Card & Account Details")
        
        mcc = st.number_input("MCC Code", min_value=0, max_value=9999, value=5411)
        credit_limit = st.number_input("Credit Limit ($)", min_value=0.0, max_value=100000.0, value=5000.0)
        card_brand = st.selectbox("Card Brand", ["Visa", "Mastercard", "American Express", "Discover"])
        card_type = st.selectbox("Card Type", ["Credit", "Debit", "Debit (Prepaid)"])
        use_chip = st.selectbox("Transaction Type", ["Chip Transaction", "Swipe Transaction", "Online Transaction"])
        has_chip = st.selectbox("Has Chip", ["YES", "NO"])
        gender = st.selectbox("Gender", ["M", "F"])
        card_on_dark_web = st.selectbox("Card on Dark Web", ["No", "Yes"])
    
    # High-risk MCC indicator
    if mcc in engine.high_risk_mccs:
        st.warning(f"⚠️ MCC {mcc} is flagged as HIGH RISK")
    
    # Submit button
    if st.button("🔍 Analyze Transaction", type="primary"):
        # Create transaction dict
        transaction = {
            'transaction_id': transaction_id,
            'user_id': user_id,
            'amount': amount,
            'currency': currency,
            'merchant': merchant,
            'timestamp': datetime.now().isoformat(),
            'location': location,
            'mcc': mcc,
            'credit_limit': credit_limit,
            'card_brand': card_brand,
            'card_type': card_type,
            'use_chip': use_chip,
            'has_chip': has_chip,
            'gender': gender,
            'card_on_dark_web': card_on_dark_web
        }
        
        # Process transaction
        with st.spinner("Analyzing transaction..."):
            result = engine.process_transaction(transaction)
        
        # Display result
        st.markdown("---")
        st.subheader("Decision Result")
        
        decision = result['decision']
        fraud_prob = result['fraud_probability']
        latency = result['latency_ms']
        
        # Decision card
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if decision == "REJECT":
                st.error(f"### 🔴 {decision}")
            elif decision == "PEND":
                st.warning(f"### 🟡 {decision}")
            else:
                st.success(f"### 🟢 {decision}")
        
        with col2:
            st.metric("Fraud Probability", f"{fraud_prob:.2%}")
        
        with col3:
            st.metric("Latency", f"{latency:.2f} ms")
        
        # Add to history
        st.session_state.decision_history.append({
            'timestamp': datetime.now(),
            'transaction_id': transaction_id,
            'amount': amount,
            'decision': decision,
            'fraud_probability': fraud_prob,
            'latency_ms': latency
        })
        
        # Display transaction details
        with st.expander("View Transaction Details"):
            st.json(transaction)

# Tab 2: Statistics
with tab2:
    st.header("Decision Statistics")
    
    if st.session_state.decision_history:
        df_history = pd.DataFrame(st.session_state.decision_history)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(df_history))
        
        with col2:
            reject_count = len(df_history[df_history['decision'] == 'REJECT'])
            st.metric("Rejected", reject_count)
        
        with col3:
            pend_count = len(df_history[df_history['decision'] == 'PEND'])
            st.metric("Pending", pend_count)
        
        with col4:
            allow_count = len(df_history[df_history['decision'] == 'ALLOW'])
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
            # Fraud probability distribution
            fig_hist = px.histogram(
                df_history,
                x='fraud_probability',
                nbins=20,
                title="Fraud Probability Distribution",
                labels={'fraud_probability': 'Fraud Probability'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Average latency
        avg_latency = df_history['latency_ms'].mean()
        st.info(f"📊 Average Decision Latency: {avg_latency:.2f} ms")
        
        # Latency over time
        fig_latency = px.line(
            df_history,
            x='timestamp',
            y='latency_ms',
            title="Decision Latency Over Time",
            labels={'timestamp': 'Time', 'latency_ms': 'Latency (ms)'}
        )
        st.plotly_chart(fig_latency, use_container_width=True)
        
    else:
        st.info("No transactions tested yet. Go to the 'Test Transaction' tab to start.")

# Tab 3: Decision History
with tab3:
    st.header("Decision History")
    
    if st.session_state.decision_history:
        df_history = pd.DataFrame(st.session_state.decision_history)
        
        # Display table
        st.dataframe(
            df_history.sort_values('timestamp', ascending=False),
            use_container_width=True,
            column_config={
                'timestamp': st.column_config.DatetimeColumn('Timestamp'),
                'amount': st.column_config.NumberColumn('Amount', format="$%.2f"),
                'fraud_probability': st.column_config.ProgressColumn('Fraud Prob', min_value=0, max_value=1),
                'latency_ms': st.column_config.NumberColumn('Latency (ms)', format="%.2f")
            }
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.decision_history = []
            st.rerun()
    else:
        st.info("No decision history available.")

# Footer
st.markdown("---")
st.markdown("### About")
st.info("""
This fraud detection system uses a GRU (Gated Recurrent Unit) neural network to analyze transactions in real-time.
The system returns one of three decisions:
- **ALLOW**: Transaction appears legitimate
- **PEND**: Transaction requires manual review
- **REJECT**: Transaction is likely fraudulent
""")