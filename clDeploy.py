import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_order.pkl")

st.title("🏦 Loan Approval Prediction")
st.write("Gradient Boosting Classification Model")

# ---- USER INPUTS ----
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)

cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# ---- CORRECT ENCODING (MATCH TRAINING) ----
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

# ---- PREDICTION ----
if st.button("Predict Loan Status"):
    input_df = pd.DataFrame([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=features)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
