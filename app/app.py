import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# TITLE
# -------------------------------
st.title("AI-Based Cardiovascular Risk Assessment System")

st.info("Note: This tool is for educational purposes only and not a medical diagnosis system.")

st.subheader("Enter Patient Details")

# -------------------------------
# INPUTS
# -------------------------------
age = st.slider("Age", 20, 80, 50)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox("Chest Pain Type", [
    "Typical Angina", 
    "Atypical Angina", 
    "Non-anginal Pain", 
    "Asymptomatic"
])

bp = st.slider("Resting Blood Pressure", 80, 200, 120)

chol = st.slider("Cholesterol", 100, 400, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])

rest_ecg = st.selectbox("Rest ECG", [
    "Normal", 
    "ST-T Abnormality", 
    "Left Ventricular Hypertrophy"
])

hr = st.slider("Max Heart Rate", 60, 200, 120)

exercise = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

# -------------------------------
# CREATE INPUT DATA
# -------------------------------
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'chest_pain_type': [cp],
    'resting_blood_pressure': [bp],
    'cholestoral': [chol],
    'fasting_blood_sugar': [fbs],
    'rest_ecg': [rest_ecg],
    'Max_heart_rate': [hr],
    'exercise_induced_angina': [exercise]
})

# -------------------------------
# ADD MISSING FEATURES (VERY IMPORTANT)
# -------------------------------
input_data['oldpeak'] = 1.0
input_data['slope'] = "Upsloping"
input_data['vessels_colored_by_flourosopy'] = "Zero"
input_data['thalassemia'] = "Normal"

# -------------------------------
# LOAD ORIGINAL DATA (FOR COLUMNS)
# -------------------------------
df = pd.read_csv('data/heart.csv')
df = pd.get_dummies(df, drop_first=True)
X = df.drop('target', axis=1)

# -------------------------------
# PROCESS INPUT
# -------------------------------
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

input_scaled = scaler.transform(input_data)

# -------------------------------
# CLEAN FEATURE NAME FUNCTION
# -------------------------------
def clean_feature_name(name):
    name = name.replace("_", " ")
    name = name.replace("cholestoral", "cholesterol")
    return name.title()

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Risk"):

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    risk_value = probability[0][1]

    st.subheader("Prediction Result")

    # Risk Level
    if risk_value > 0.7:
        st.error("High Risk of Cardiovascular Disease")
    elif risk_value > 0.4:
        st.warning("Moderate Risk of Cardiovascular Disease")
    else:
        st.success("Low Risk of Cardiovascular Disease")

    st.metric("Risk Probability", f"{risk_value*100:.2f}%")

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    st.subheader("Risk Visualization")
    st.progress(int(risk_value * 100))

    # -------------------------------
    # TOP FACTORS
    # -------------------------------
    st.subheader("Top Factors Influencing Risk")

    importances = model.feature_importances_
    features = X.columns

    indices = np.argsort(importances)[::-1][:5]

    for i in indices:
        clean_name = clean_feature_name(features[i])
        st.markdown(f"- **{clean_name}**")

    # -------------------------------
    # INTERPRETATION
    # -------------------------------
    st.subheader("Clinical Interpretation")

    if risk_value > 0.7:
        st.write("""
        - Strong indicators of cardiovascular stress detected  
        - Immediate clinical evaluation is recommended  
        - Risk factors suggest possible cardiac complications  
        """)
    elif risk_value > 0.4:
        st.write("""
        - Moderate risk detected based on input parameters  
        - Lifestyle changes and monitoring are advised  
        """)
    else:
        st.write("""
        - No major cardiovascular risk patterns detected  
        - Maintain current health conditions  
        """)

    # -------------------------------
    # LIMITATIONS
    # -------------------------------
    st.warning("""
    Limitations:
    - Based on limited dataset (UCI)
    - Not a substitute for clinical diagnosis
    - Requires validation on real patient data
    """)