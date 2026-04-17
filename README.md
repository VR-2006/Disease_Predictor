# Disease Risk Predictor

## Overview

This project focuses on predicting the likelihood of heart disease using machine learning techniques. It includes data analysis, preprocessing, model development, and performance evaluation.

The objective is to build an accurate and interpretable system for early disease risk assessment.

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

## Project Workflow

### Day 1 — Exploratory Data Analysis (EDA)

* Loaded and explored the dataset
* Examined dataset structure, features, and statistical properties
* Verified absence of missing values
* Performed data visualization:

  * Target distribution
  * Age distribution
  * Age vs Disease relationship
  * Feature correlation heatmap

#### Key Insights

* Dataset is balanced with respect to target classes
* Majority of individuals fall within the 45–65 age group
* Age shows a strong relationship with disease presence
* Certain features (e.g., chest pain type, heart rate) exhibit strong correlation

---

### Day 2 — Data Preprocessing

* Converted categorical variables into numeric format using one-hot encoding
* Separated features (X) and target variable (y)
* Performed train-test split (80% training, 20% testing)
* Applied feature scaling using StandardScaler

#### Outcome

* Prepared a clean and structured dataset suitable for machine learning models
* Established a preprocessing pipeline

---

### Day 3 — Model Development

* Implemented Logistic Regression
* Trained model on training dataset
* Evaluated using:

  * Accuracy score
  * Classification report

#### Results

* Achieved approximately 82% accuracy
* High recall for disease detection (~88%)
* Balanced performance across classes

#### Insight

* Model effectively identifies disease cases, which is important for medical applications

---

### Day 4 — Model Improvement and Evaluation

* Implemented Random Forest Classifier
* Compared performance with Logistic Regression
* Evaluated using:

  * Accuracy comparison
  * Confusion matrix
  * Classification report

## Day 6 - Web Application

- Developed an interactive web app using Streamlit  
- Enabled real-time disease prediction based on user input  
- Integrated trained Random Forest model  

## Day 7 - Final Enhancements

- Improved UI and usability  
- Added prediction explanation  
- Structured project for deployment  

## Final Outcome
- End-to-end machine learning pipeline  
- Interactive application for real-time prediction  
- Strong project suitable for portfolio and resume    

#### Improvements

* Random Forest captured more complex patterns
* Improved robustness of predictions
* Enhanced overall model performance

#### Evaluation Insight

* Confusion matrix provided detailed understanding of correct and incorrect classifications

---

## Current Status

* Data exploration completed
* Data preprocessing pipeline established
* Baseline and improved models implemented
* Model performance evaluated

---

## Next Steps

* Feature importance analysis
* Model interpretability enhancements
* Deployment using a web interface

---

## Project Structure

```bash
Disease_Predictor/
│
├── data/
│   └── heart.csv
│
├── notebooks/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── model_improvement.py
│
├── README.md
├── requirements.txt
```

---

## Key Takeaway

This project demonstrates a complete machine learning pipeline, from raw data processing to model evaluation and improvement.

---

## Author

Vignesh R
Biomedical Engineering Student
