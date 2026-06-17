# 📊 Telecom Customer Churn Prediction

A machine learning web application that predicts whether a telecom customer is likely to churn based on customer demographics, account information, and subscribed services.

## 🚀 Live Demo

https://customer-churn-xgb.streamlit.app

## 📌 Project Overview

Customer churn is a major challenge for telecom companies. This project uses an XGBoost classification model to identify customers who are likely to leave the service.

The application allows users to enter customer details through an interactive Streamlit interface and instantly receive a churn prediction.

## 🎯 Problem Statement

Predict whether a telecom customer will churn based on:

- Demographics
- Account information
- Services subscribed
- Billing information

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Git & GitHub

## 📂 Dataset

IBM Telco Customer Churn Dataset

Features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Contract Type
- Internet Service
- Online Security
- Tech Support
- Monthly Charges
- Total Charges
- Payment Method

Target Variable:

- Churn (Yes / No)

## ⚙️ Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Feature encoding
- Feature scaling
- ColumnTransformer pipeline

### Feature Engineering

Created:

- AvgMonthlySpend
- IsNewCustomer

### Model

- XGBoost Classifier

### Hyperparameter Tuning

- RandomizedSearchCV
- Cross Validation

## 📈 Model Performance

| Metric | Score |
|----------|----------|
| ROC-AUC | 0.76 |
| Recall | 0.87 |
| Precision | 0.47 |
| F1 Score | 0.61 |

*Values may vary slightly depending on model version.*

## 🌐 Streamlit Application

The application allows users to:

- Enter customer information
- View customer summary
- Predict churn probability
- Get instant churn prediction

## 📸 Screenshots

Add screenshots here.

Example:

![App Screenshot](screenshot.png)

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/shivbrxt/ml_projects.git
```

Move into project folder:

```bash
cd ml_projects
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app1.py
```

## 📁 Repository Structure

```text
ml_projects/
│
├── app1.py
├── telecom_customer_churn_model_XGB.pkl
├── requirements.txt
└── README.md
```

## 🔮 Future Improvements

- Probability-based predictions
- Model explainability using SHAP
- Deployment on AWS
- Customer churn visualization dashboard

## 👨‍💻 Author

Shiv Gajsingh

MBA Marketing | Aspiring Data Analyst

GitHub:
https://github.com/shivbrxt

LinkedIn:
(Add your LinkedIn profile link)
