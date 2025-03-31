import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "Processed_Customer_Churn.csv"  # Update the correct path
data = pd.read_csv(file_path)

# Display column names
print("Available columns:", data.columns)

# Identify categorical columns (ensure they exist)
categorical_cols = [col for col in ['gender', 'country'] if col in data.columns]

# Apply one-hot encoding only if categorical columns exist
if categorical_cols:
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=['churn'])  # Ensure 'churn' column exists
y = data['churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_test_pred = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

# Performance Metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
auc_roc = roc_auc_score(y_test, y_test_prob)

print("\n✅ Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")

# Feature Importance Plot
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances['Importance'], y=feature_importances['Feature'], palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.show()

# Save results
data['Churn_Probability'] = model.predict_proba(X)[:, 1]
data.to_csv("Churn_Predictions.csv", index=False)
print("\n🔹 Churn predictions saved in 'Churn_Predictions.csv'!")
