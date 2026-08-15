# Telco Customer Churn Classification Models App

This repository contains 5 Machine Learning classification models trained on the Telco Customer Churn dataset from Kaggle. It includes an interactive Streamlit application to display evaluation metrics and make predictions.

## Problem Statement
For any telecommunication company, retaining existing customers is significantly more important than acquiring new ones because it is cost-effective. The term customer churn refers to a loss of existing customer.

With this project we are building and evaluating multiple machine learning classification models on there correctness to predict whether a customer will churn or stay. This will be done by analyzing customer data such as demographics, subscribed services, and account information etc., this binary classification system will in return help to identify "at-risk" customers. This will allow the business to proactively implement targeted retention strategies, improve customer satisfaction, and minimize revenue loss.

## Dataset Description
This project is using the Telco Customer Churn dataset sourced from Kaggle. The dataset provides a comprehensive view of customer profiles, their usage patterns, and billing information.

- Total Instances: 7,043
- Total Features: 20 predictor variables
- Target Variable: Churn (Binary: "Yes" or "No")

Feature Categories:

1. Demographics: Information about the customer, including gender, whether they are a senior citizen, and if they have a partner or dependents.

2. Services Subscribed: Specific services the customer has signed up for, such as phone service, multiple lines, internet service type (DSL/Fiber optic), online security, online backup, device protection, tech support, and streaming TV/movies.

3. Account Information: Details regarding the customer's account, including their tenure (number of months with the company), contract type (Month-to-month, One year, Two year), paperless billing status, payment method, monthly charges, and total accumulated charges.

## Implemented Models
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes (Gaussian)
- Random Forest

## Evaluation metrics results
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.785359 | 0.830738 | 0.620805 | 0.494652 | 0.550595 | 0.416551 |
| **Decision Tree** | 0.788202 | 0.812496 | 0.642857 | 0.457219 | 0.534375 | 0.412095 |
| **kNN** | 0.739872 | 0.756661 | 0.510753 | 0.508021 | 0.509383 | 0.332405 |
| **Naive Bayes** | 0.737740 | 0.812689 | 0.504638 | 0.727273 | 0.595838 | 0.426012 |
| **Random Forest (Ensemble)** | 0.788202 | 0.811790 | 0.634752 | 0.478610 | 0.545732 | 0.418129 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall AUC score (0.830738), indicating superior class separation capability. It offers strong balanced performance across accuracy and precision. |
| **Decision Tree** | Tied for highest accuracy (0.788202) and highest precision (0.642857). However, low recall (0.457219) means it misses more than 54% of actual customer churn cases. |
| **kNN** | The weakest overall model across most metrics (AUC: 0.756661, MCC: 0.332405). Highly sensitive to feature density and distance metrics despite standardization. |
| **Naive Bayes** | Highest Recall (0.727273), F1 Score (0.595838), and MCC (0.426012). Highly effective at minimizing False Negatives by detecting nearly 73% of all churning customers. |
| **Random Forest (Ensemble)** | Tied with Decision Tree for highest accuracy (0.788202) while showing improved recall (0.478610) and F1 score (0.545732). Offers high stability and lower variance. |
| **Overall Winner for your dataset?** | **Naive Bayes** is the business winner for customer churn, as high **Recall (0.727273)** ensures the company identifies the maximum number of churning customers. Alternatively, **Logistic Regression** serves as the best general-purpose model with the highest **AUC Score (0.830738)**. |

## How to run locally
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
