# breast-cancer-predictor
# 🧬 Breast Cancer Prediction App

🚀 A Machine Learning web app that predicts whether a tumor is **Benign (Safe)** or **Malignant (Cancer)** using Logistic Regression.

---

## 🌍 Live Demo

👉 **Try the app here:**
 https://cancer-predictor-anurag.streamlit.app/

---

## 🧠 Project Overview

This project uses a trained **Logistic Regression model** to classify breast tumors based on medical features.

Users can input tumor characteristics and get:

* ✅ Prediction (Benign / Malignant)
* 📊 Probability score
* 💡 Basic explanation

---

## ⚙️ Tech Stack

* 🐍 Python
* 🤖 Scikit-learn
* 📊 Pandas
* 🌐 Streamlit
* 💾 Joblib

---

## 📊 Features Used

* texture_worst
* radius_mean
* area_worst
* concavity_worst
* symmetry_worst

---

## 🏗️ Project Structure

```bash
breast-cancer-predictor/
├── app.py                  # Streamlit app
├── cancer_model.pkl        # Trained ML model
├── requirements.txt        # Dependencies
├── model_training.ipynb    # Model training notebook
```

---

## 🔬 Model Details

* Algorithm: Logistic Regression
* Preprocessing: StandardScaler
* Output: Probability using sigmoid function

---

## 🎯 How It Works

1. User enters tumor data
2. Data is processed and scaled
3. Model predicts probability
4. Output shown as:

   * ⚠️ Malignant
   * ✅ Benign

---

## 🧪 Example Inputs

### 🟢 Benign

* texture_worst = 12
* radius_mean = 11
* area_worst = 400
* concavity_worst = 0.03
* symmetry_worst = 0.18

---

### 🔴 Malignant

* texture_worst = 35
* radius_mean = 20
* area_worst = 1200
* concavity_worst = 0.4
* symmetry_worst = 0.35

---

## 🚀 Deployment

Deployed using **Streamlit Cloud**

---

## 💼 Use Case

* Medical data analysis (educational use)
* ML model deployment practice
* Beginner-friendly end-to-end project

---

## ⚠️ Disclaimer

This app is for **educational purposes only** and should not be used for real medical diagnosis.

---

## 👨‍💻 Author

**Anurag**
ML Enthusiast 🚀

---
