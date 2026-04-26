
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("cancer_model.pkl")
features = model.feature_names_in_

st.title("🧬 Breast Cancer Prediction App")

st.write("Enter patient details:")

 
# Select only important features
important_features = [
    "texture_worst",
    "radius_mean",
    "area_worst",
    "concavity_worst",
    "symmetry_worst"
]

input_data = {}

# Create sliders for important features
for feature in important_features:
    input_data[feature] = st.slider(feature, 0.0, 50.0, 10.0)

# Fill remaining features with 0
for feature in features:
    if feature not in input_data:
        input_data[feature] = 0
 


# Predict
if st.button("Predict"):
  
   df = pd.DataFrame([input_data])

# 🔥 IMPORTANT FIX: match training feature order
   df = df[features]
 


    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Malignant (Cancer) | Probability: {probability:.2f}")
    else:
        st.success(f"✅ Benign (Safe) | Probability: {probability:.2f}")
 
