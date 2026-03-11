import streamlit as st
import requests
import time
import pandas as pd
import uuid

st.set_page_config(page_title="Fraud Command Center", page_icon="🛡️", layout="wide")
API_URL = "http://localhost:8000/api/v1"


SCENARIOS = {
    "🟢 Legit Grocery Run": {
        "Time": 120.0, "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33,
        "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09, "V11": -0.55,
        "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
        "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25, "V21": -0.01, "V22": 0.27,
        "V23": -0.11, "V24": 0.06, "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02,
        "Amount": 45.50
    },
    "🟡 Subtle Anomaly": {
        "Time": 150.0, "V1": -0.5, "V2": 0.8, "V3": -0.2, "V4": 2.1, "V5": -0.1,
        "V6": -0.5, "V7": 0.1, "V8": 0.2, "V9": -1.0, "V10": -0.8, "V11": 1.2,
        "V12": -1.5, "V13": -0.2, "V14": -2.0, "V15": 0.5, "V16": -0.8,
        "V17": -1.2, "V18": 0.1, "V19": 0.3, "V20": 0.1, "V21": 0.2, "V22": 0.1,
        "V23": -0.1, "V24": 0.2, "V25": 0.1, "V26": 0.3, "V27": 0.1, "V28": -0.1,
        "Amount": 250.00
    },
    "🔴 Stolen Card Attack": {
        "Time": 406.0, "V1": -2.31, "V2": 1.95, "V3": -1.60, "V4": 3.99, "V5": -0.52,
        "V6": -1.42, "V7": -2.53, "V8": 1.39, "V9": -2.77, "V10": -2.77, "V11": 3.20,
        "V12": -2.89, "V13": -0.59, "V14": -4.28, "V15": 0.38, "V16": -1.14,
        "V17": -2.83, "V18": -0.01, "V19": 0.41, "V20": 0.12, "V21": 0.51, "V22": -0.03,
        "V23": -0.46, "V24": 0.32, "V25": 0.04, "V26": 0.17, "V27": 0.26, "V28": -0.14,
        "Amount": 0.00
    }
}

st.title("🛡️ Enterprise Fraud Command Center")
st.markdown("Monitor real-time transactions processed by the dual-stage ML pipeline.")

with st.sidebar:
    st.header("1. Select Scenario")
    selected_scenario = st.radio("Choose a transaction type:", list(SCENARIOS.keys()))
    
    st.header("2. Cache Settings")
    use_cached_id = st.checkbox("Simulate Network Retry (Cache Hit)", value=False, 
                                help="Keeps the same transaction ID. The second time you scan, Redis will intercept it.")
    
    st.markdown("---")
    st.markdown("**Architecture:**\n* Stage 1: Isolation Forest\n* Stage 2: XGBoost\n* Cache: Redis\n* DB: PostgreSQL")

payload = SCENARIOS[selected_scenario].copy()

if "current_txn_id" not in st.session_state or not use_cached_id:
    st.session_state.current_txn_id = f"txn-{uuid.uuid4().hex[:8]}"
payload["transaction_id"] = st.session_state.current_txn_id

# UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Incoming Payload")
    st.json(payload)
    scan_btn = st.button("🔍 Scan Transaction", type="primary", use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    if scan_btn:
        with st.spinner("Analyzing via Cascade Pipeline..."):
            start_time = time.time()
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is Docker running?")
                st.stop()
                
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Label", data["risk_label"])
                m2.metric("Fraud Probability", f"{data['fraud_probability']:.2%}")
                
                if latency_ms < 15:
                    m3.metric("Latency", f"{latency_ms:.1f} ms", "⚡ Cache Hit")
                else:
                    m3.metric("Latency", f"{latency_ms:.1f} ms", "DB Write", delta_color="off")

                if data["risk_label"] in ["HIGH", "MEDIUM"]:
                    st.warning("⚠️ Transaction flagged. Fetching model reasoning...")
                    
                    exp_resp = requests.post(f"{API_URL}/explain", json=payload)
                    if exp_resp.status_code == 200:
                        exp_data = exp_resp.json()
                        st.markdown(f"**Triggered at Pipeline Stage:** `{exp_data['processing_stage']}`")
                        
                        st.markdown("### Model's Argument (SHAP Values)")
                        
                        features = [f["feature"] for f in exp_data["top_features"]]
                        impacts = [f["shap_value"] for f in exp_data["top_features"]]
                        chart_df = pd.DataFrame({"Feature": features, "Impact": impacts}).set_index("Feature")
                        
                        st.bar_chart(chart_df, color="#ff4b4b")
                        
            elif response.status_code == 409:
                st.error("🛑 409 Conflict: This exact transaction ID was already processed and saved to the database. (Uncheck the 'Simulate Network Retry' box to generate a new ID).")
            else:
                st.error(f"API Error: {response.status_code}")