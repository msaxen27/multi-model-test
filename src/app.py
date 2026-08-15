import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="ML model Classification App", layout="wide")

st.title("Machine Learning model Classification App")
st.markdown("Demonstrating 5 Machine Learning Models on the Telco Customer Churn Dataset.")

# Load Test Data and Metrics
@st.cache_data
def load_data():
    test_df = pd.read_csv('test_data.csv')
    metrics_df = pd.read_csv('model/metrics.csv')
    return test_df, metrics_df

try:
    test_data, metrics_df = load_data()
except FileNotFoundError:
    st.error("Data files not found. Run the training script first to generate the necessary files.")
    st.stop()

# Sidebar: Select the Model from the list of available models
st.sidebar.header("Configuration")
model_names = metrics_df['Model'].tolist()
selected_model = st.sidebar.selectbox("Choose a Classification Model", model_names)

# 1. Display Metrics for the selected model
st.header(f"Performance Metrics: {selected_model}")
model_metrics = metrics_df[metrics_df['Model'] == selected_model].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
col2.metric("AUC Score", f"{model_metrics['AUC Score']:.4f}")
col3.metric("Precision", f"{model_metrics['Precision']:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("Recall", f"{model_metrics['Recall']:.4f}")
col5.metric("F1 Score", f"{model_metrics['F1 Score']:.4f}")
col6.metric("MCC Score", f"{model_metrics['MCC Score']:.4f}")

st.divider()

# 2. Interactive Prediction
st.header("Interactive Prediction")
st.write("Because this dataset has 19 features after preprocessing, selecting a sample from our test data is the easiest way to test the models interactively.")

# Slider to pick a row from test_data
sample_index = st.slider("Select a Row Index from Test Data", 0, len(test_data) - 1, 0)
sample = test_data.iloc[[sample_index]].drop('target', axis=1)
actual_label = test_data.iloc[sample_index]['target']

st.write("**Features for selected sample (Scaled):**")
st.dataframe(sample)

# Load the corresponding saved model
model_filename = selected_model.replace(" ", "_").lower() + '.pkl'
model_path = os.path.join('model', model_filename)

if st.button("Predict"):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        prediction = model.predict(sample)[0]
        
        # In this dataset: 0 = No Churn, 1 = Churn
        pred_class = "Churn" if prediction == 1 else "No Churn"
        actual_class = "Churn" if actual_label == 1 else "No Churn"
        
        st.subheader("Results")
        if prediction == actual_label:
            st.success(f"✅ Correct Prediction! The model predicted: **{pred_class}**")
        else:
            st.error(f"❌ Incorrect Prediction. The model predicted: **{pred_class}** (Actual: **{actual_class}**)")
    else:
        st.error(f"Model file {model_filename} not found in 'model/' directory.")