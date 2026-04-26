 
import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Cancer Predictor", layout="centered")

# Load model
model = joblib.load("cancer_model.pkl")
features = model.feature_names_in_

# Title
st.title("🧬 Breast Cancer Prediction App")
st.write("Enter key tumor features to predict whether it is benign or malignant.")

# Important features (clean UI)
important_features = [
    "texture_worst",
    "radius_mean",
    "area_worst",
    "concavity_worst",
    "symmetry_worst"
]
 
# Sliders with correct ranges
input_data = {}

input_data["texture_worst"] = st.slider("texture_worst", 0.0, 50.0, 10.0)
input_data["radius_mean"] = st.slider("radius_mean", 0.0, 30.0, 10.0)
input_data["area_worst"] = st.slider("area_worst", 0.0, 2000.0, 500.0)
input_data["concavity_worst"] = st.slider("concavity_worst", 0.0, 1.0, 0.1)
input_data["symmetry_worst"] = st.slider("symmetry_worst", 0.0, 1.0, 0.2)
 
 

 

# Fill remaining features with 0
for feature in features:
    if feature not in input_data:
        input_data[feature] = 0

# Predict button
if st.button("Predict"):
    df = pd.DataFrame([input_data])

    # 🔥 VERY IMPORTANT: match training feature order
    df = df[features]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    st.write("### Result")

    if prediction == 1:
        st.error(f"⚠️ Malignant (Cancer) | Probability: {probability:.2f}")
    else:
        st.success(f"✅ Benign (Safe) | Probability: {probability:.2f}")

    # Confidence bar
    st.write("### Prediction Confidence")
    st.progress(float(probability))

    # Simple explanation
    st.write("### Insight")
    if prediction == 1:
        st.write("Higher texture, area, and irregular shape indicate higher cancer risk.")
    else:
        st.write("Lower values suggest a smoother and less irregular tumor (likely benign).")

 
