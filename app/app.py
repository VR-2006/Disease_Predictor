import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Title
st.title("Disease Risk Predictor")
st.subheader("Simple AI-based Health Risk Assessment Tool")
st.write("Enter patient details to predict disease risk")

# Inputs
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
bp = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
hr = st.slider("Max Heart Rate", 60, 200, 120)

# Convert input to dataframe
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'resting_blood_pressure': [bp],
    'cholestoral': [chol],
    'Max_heart_rate': [hr]
})

# Load dataset and preprocess
df = pd.read_csv('data/heart.csv')
df = pd.get_dummies(df, drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Align input with training columns
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("High Risk of Disease")
        st.write("Consult a medical professional.")
    else:
        st.success("Low Risk of Disease")
        st.write("Maintain a healthy lifestyle.")