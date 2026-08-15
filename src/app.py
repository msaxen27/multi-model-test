import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="ML model Classification App", layout="wide")

st.title("Machine Learning model Classification App")
st.markdown("Demonstrating 5 Machine Learning Models on the Telco Customer Churn Dataset.")

# Load Test Data and Metrics (paths resolved relative to this file)
BASE_DIR = os.path.dirname(__file__)


@st.cache_data
def load_data():
    test_path = os.path.join(BASE_DIR, 'test_data.csv')
    metrics_path = os.path.join(BASE_DIR, 'model', 'metrics.csv')
    test_df = pd.read_csv(test_path)
    metrics_df = pd.read_csv(metrics_path)
    return test_df, metrics_df

try:
    test_data, metrics_df = load_data()
except FileNotFoundError as e:
    st.error("Data files not found. Run the training script first to generate the necessary files.")
    st.error(f"Attempted paths:\n test_data: {os.path.join(BASE_DIR, 'test_data.csv')}\n metrics: {os.path.join(BASE_DIR, 'model', 'metrics.csv')}")
    st.stop()

# Sidebar: Select the Model from the list of available models
st.sidebar.header("Configuration")
model_names = metrics_df['Model'].tolist()
selected_model = st.sidebar.selectbox("Choose a Classification Model", model_names)

# 1. Display Metrics for the selected model
st.header(f"Performance Metrics: {selected_model}")
model_metrics = metrics_df[metrics_df['Model'] == selected_model].iloc[0]

# Render metrics as circular translucent green bubbles using inline CSS
bubble_css = """
<style>
.metrics-row {display:flex; gap:18px; margin:16px 0 8px 0;}
.metric-bubble {
    width:140px; height:140px; border-radius:50%;
    background: rgba(16,185,129,0.12);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    box-shadow: 0 6px 18px rgba(6,95,70,0.08);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.metric-value { font-size:20px; font-weight:700; color:#ffffff; }
.metric-label { font-size:12px; color:#F54927; opacity:0.95; margin-top:6px; }
@media (max-width:700px){ .metrics-row {flex-wrap:wrap; justify-content:center;} .metric-bubble{width:110px;height:110px;} }
</style>
"""

st.markdown(bubble_css, unsafe_allow_html=True)

widget_css = """
<style>
/* Predict button */
.stButton > button {
  background: linear-gradient(180deg,#3b82f6,#2563eb) !important;
  color: #ffffff !important;
  border: none !important;
  box-shadow: 0 6px 18px rgba(37,99,235,0.18) !important;
  border-radius: 8px !important;
  padding: 8px 18px !important;
  font-weight:600 !important;
}
.stButton > button:hover {
  filter: brightness(0.95) !important;
}

/* Sidebar selectbox dropdown */
.stSelectbox > div > div,
[data-baseweb="select"] > div,
[data-baseweb="select"] > div > div {
  background: linear-gradient(180deg, #3D6B8F 0%, #82B3AA 100%) !important;
  border: 1px solid #60a5fa !important;
  border-radius: 8px !important;
  color: #0f172a !important;
  box-shadow: 0 4px 12px rgba(37,99,235,0.10) !important;
}

.stSelectbox > div > div > div,
[data-baseweb="select"] span,
[data-baseweb="select"] div {
  color: #0f172a !important;
}

/* Ensure width looks good */
.stSelectbox > div > div {
  width: 220px !important;
}

/* Force visible blue selected area */
div[data-baseweb="select"] > div[role="button"] {
  background: linear-gradient(180deg, #3D6B8F 0%, #82B3AA 100%) !important;
}
</style>
"""

st.markdown(widget_css, unsafe_allow_html=True)

acc = f"{model_metrics['Accuracy']:.4f}"
auc = f"{model_metrics['AUC Score']:.4f}"
prec = f"{model_metrics['Precision']:.4f}"
rec = f"{model_metrics['Recall']:.4f}"
f1 = f"{model_metrics['F1 Score']:.4f}"
mcc = f"{model_metrics['MCC Score']:.4f}"

metrics_html = f"""
<div class="metrics-row">
    <div class="metric-bubble">
        <div class="metric-value">{acc}</div>
        <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric-bubble">
        <div class="metric-value">{auc}</div>
        <div class="metric-label">AUC Score</div>
    </div>
    <div class="metric-bubble">
        <div class="metric-value">{prec}</div>
        <div class="metric-label">Precision</div>
    </div>
</div>
<div class="metrics-row">
    <div class="metric-bubble">
        <div class="metric-value">{rec}</div>
        <div class="metric-label">Recall</div>
    </div>
    <div class="metric-bubble">
        <div class="metric-value">{f1}</div>
        <div class="metric-label">F1 Score</div>
    </div>
    <div class="metric-bubble">
        <div class="metric-value">{mcc}</div>
        <div class="metric-label">MCC Score</div>
    </div>
</div>
"""

st.markdown(metrics_html, unsafe_allow_html=True)

st.divider()

# 2. Interactive Prediction
st.header("Churn Prediction")
st.write("Select a sample from the test data to check different models predictions interactively.")

# Slider to pick a row from test_data
sample_index = st.slider("Select a Row Index from Test Data", 0, len(test_data) - 1, 0)
sample = test_data.iloc[[sample_index]].drop('target', axis=1)
actual_label = test_data.iloc[sample_index]['target']

st.write("**Features for selected sample (Scaled):**")
st.dataframe(sample)

# Load the corresponding saved model
model_filename = selected_model.replace(" ", "_").lower() + '.pkl'
model_path = os.path.join(BASE_DIR, 'model', model_filename)

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