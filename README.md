# AI-Based Cardiovascular Risk Assessment System

## Overview

This project presents a machine learning-based system designed to assess cardiovascular disease risk using clinical parameters.

It integrates data preprocessing, model training, and an interactive Streamlit web application to provide real-time predictions along with clinical interpretation and visualization.

The system simulates a basic clinical decision-support tool for early risk assessment.

---

## Features

* Risk classification (Low / Moderate / High)
* Probability-based risk estimation
* Clinical interpretation of results
* Feature importance analysis
* Risk visualization using progress bar
* Interactive user interface using Streamlit

---

## Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Streamlit

---

## Project Structure

Disease_Predictor/
├── app/
│   └── app.py
├── data/
│   └── heart.csv
├── notebooks/
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── app_screenshot.png

---

## Example Usage

### Low Risk Input

* Age: 30
* Sex: Female
* Chest Pain Type: Non-anginal Pain
* Resting Blood Pressure: 110 mmHg
* Cholesterol: 180 mg/dl
* Fasting Blood Sugar: No
* Rest ECG: Normal
* Max Heart Rate: 170 bpm
* Exercise Induced Angina: No

Expected Output: **Low Risk of Cardiovascular Disease**

---

### High Risk Input

* Age: 60
* Sex: Male
* Chest Pain Type: Asymptomatic
* Resting Blood Pressure: 160 mmHg
* Cholesterol: 300 mg/dl
* Fasting Blood Sugar: Yes
* Rest ECG: ST-T Abnormality
* Max Heart Rate: 100 bpm
* Exercise Induced Angina: Yes

Expected Output: **High Risk of Cardiovascular Disease**

---

## Note

Default values are used for certain clinical parameters (oldpeak, slope, vessels, thalassemia) to ensure stable predictions due to dataset limitations.

---

## Limitations

* Based on a limited dataset (UCI Heart Disease Dataset)
* Not clinically validated
* Intended for educational and demonstration purposes only

---

## Application Preview

![App Screenshot](app_screenshot.png)

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Author

Vignesh Ram R
Biomedical Engineering Student
