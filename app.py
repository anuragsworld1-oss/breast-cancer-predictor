
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("cancer_model.pkl")
features = model.feature_names_in_

st.title("🧬 Breast Cancer Prediction App")

st.write("Enter patient details:")

# Input fields
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(feature, value=0.0)

# Predict
if st.button("Predict"):
    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Malignant (Cancer) | Probability: {probability:.2f}")
    else:
        st.success(f"✅ Benign (Safe) | Probability: {probability:.2f}")
 
