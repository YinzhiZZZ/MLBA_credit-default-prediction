import streamlit as st
import joblib
import numpy as np
import boto3
import pandas as pd
import io
from datetime import datetime

st.set_page_config(page_title="Credit Default Risk Predictor", page_icon="💳", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('/home/ubuntu/MLBA_credit-default-prediction/models/xgboost_model.pkl')

def save_to_dynamodb(record):
    try:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.Table('credit-predictions')
        table.put_item(Item=record)
        return True
    except Exception as e:
        return False

st.title("💳 Credit Card Default Risk Predictor")
st.markdown("Enter customer information to predict default probability.")

tab1, tab2 = st.tabs(["Single Customer", "Batch Prediction (S3)"])

with tab1:
    st.subheader("Customer Profile")
    col1, col2 = st.columns(2)
    with col1:
        limit_bal = st.slider("Credit Limit (NT$)", 10000, 1000000, 50000, step=10000)
        age = st.slider("Age", 18, 75, 30)
        sex = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        education = st.selectbox("Education", ["University", "Graduate School", "High School", "Other"])
        marriage = st.selectbox("Marital Status", ["Single", "Married", "Other"])

    st.subheader("Payment Status (last 3 months)")
    st.caption("-2: No use | -1: Paid in full | 1~8: Months delayed")
    pay0 = st.slider("Pay Status Sep", -2, 8, -1)
    pay2 = st.slider("Pay Status Aug", -2, 8, -1)
    pay3 = st.slider("Pay Status Jul", -2, 8, -1)

    st.subheader("Bill & Payment Amounts (NT$)")
    col3, col4 = st.columns(2)
    with col3:
        bill1 = st.number_input("Bill Amount Sep", value=3913)
        bill2 = st.number_input("Bill Amount Aug", value=3102)
    with col4:
        amt1 = st.number_input("Payment Sep", value=0)
        amt2 = st.number_input("Payment Aug", value=689)

    if st.button("Predict Default Risk", type="primary"):
        model = load_model()
        sex_val = 2 if sex == "Female" else 1
        edu_val = {"Graduate School":1,"University":2,"High School":3,"Other":4}[education]
        mar_val = {"Married":1,"Single":2,"Other":3}[marriage]
        X = np.array([[limit_bal, sex_val, edu_val, mar_val, age,
                       pay0, pay2, pay3, -1, -1, -1,
                       bill1, bill2, 0, 0, 0, 0,
                       amt1, amt2, 0, 0, 0, 0]])
        prob = float(model.predict_proba(X)[0][1])
        risk = "HIGH RISK" if prob > 0.5 else "LOW RISK"
        st.divider()
        if prob > 0.5:
            st.error(f"⚠️ {risk}")
        else:
            st.success(f"✅ {risk}")
        st.metric("Default Probability", f"{prob:.1%}")
        st.progress(prob)
        record = {
            'id': str(datetime.now().timestamp()),
            'timestamp': str(datetime.now()),
            'source': 'streamlit_ec2',
            'credit_limi
