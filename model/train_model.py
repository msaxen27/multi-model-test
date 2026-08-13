import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('ggplot')

# 1. Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
#df.head()

print("\n--- Data Info & Types ---")
display(df.info())

display("Statistical Summary for all Numerical Columns")
display(df.describe())

# 2. Preprocessing
# Drop customerID as it's not a useful feature
df.drop('customerID', axis=1, inplace=True)
#df.head()

# Convert TotalCharges to numeric, replacing errors with NaN, then drop NaNs
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
#df.head()s

# Encode Categorical Variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale numerical features (important for KNN and Logistic Regression)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Save the scaler and label encoders for the Streamlit app
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the test set for our Streamlit app
test_data = X_test.copy()
test_data['target'] = y_test.values
test_data.to_csv('test_data.csv', index=False)

# 4. Initialize Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42), # Added max_depth to prevent overfitting
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Dictionaries to store predictions for visualizations
y_preds = {}
y_probs = {}
results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Store predictions for plotting
    y_preds[name] = y_pred
    y_probs[name] = y_prob
    
    # Calculate Metrics
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC Score": matthews_corrcoef(y_test, y_pred)
    }
    results.append(metrics)
    
    # Save Model as a .pkl file inside the 'model' directory
    filename = f"model/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"Saved: {filename}")

results_df = pd.DataFrame(results)
results_df.to_csv('model/metrics.csv', index=False)
print("\nAll models trained and evaluated successfully!")
print(results_df)

# Plot Confusion Matrix for all models
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# axes = axes.flatten()

# for i, (name, y_pred) in enumerate(y_preds.items()):
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    
#     disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
#     axes[i].set_title(f'{name} Confusion Matrix')
#     axes[i].grid(False)

# # Hide the 6th empty subplot (since we only have 5 models)
# axes[5].axis('off')

# plt.tight_layout()
# plt.show()

# # Plot ROC Curve for all models on a single graph
# plt.figure(figsize=(10, 8))

# for name, y_prob in y_probs.items():
#     fpr, tpr, thresholds = roc_curve(y_test, y_prob)
#     auc = roc_auc_score(y_test, y_prob)
#     plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

# # Plot the random classifier diagonal line
# plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', linewidth=2)

# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate (FPR)', fontsize=12)
# plt.ylabel('True Positive Rate (TPR)', fontsize=12)
# plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=15)
# plt.legend(loc="lower right", fontsize=11)
# plt.grid(alpha=0.3)
# plt.show()