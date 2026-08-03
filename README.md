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

## How to run locally
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
