# Import necessary libraries
import pandas as pd
import numpy as np
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Upload dataset (for Google Colab users)
uploaded = files.upload()

# Load the dataset
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)

# Display the first few rows
print("\n📊 First 5 rows of the dataset:")
print(data.head())

# Check available columns
print("\n✅ Available columns in dataset:", list(data.columns))

# Handle missing categorical columns
if 'gender' in data.columns:
    data['gender'] = LabelEncoder().fit_transform(data['gender'])
else:
    print("⚠️ Warning: 'gender' column not found in dataset.")

if 'country' in data.columns:
    data = pd.get_dummies(data, columns=['country'], drop_first=True)
else:
    print("⚠️ Warning: 'country' column not found in dataset.")

# Define features and target
features = ['credit_score', 'age', 'balance', 'products_number', 'estimated_salary', 'active_member']
if 'gender' in data.columns:
    features.append('gender')

X = data[features]
y = data['churn']  # Assuming the target column is named 'churn'

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model for feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Get feature importance scores
feature_importance = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette="viridis")
plt.title("🔍 Key Churn Factors (Feature Importance)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Predict churn probabilities
y_test_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Segment at-risk customers (probability > 0.5)
test_results = X_test.copy()
test_results['Churn_Probability'] = y_test_prob
test_results['Actual_Churn'] = y_test.values
at_risk_customers = test_results[test_results['Churn_Probability'] > 0.5]

# Display top at-risk customers
print("\n🚨 At-Risk Customers (High Churn Probability):")
print(at_risk_customers.sort_values(by='Churn_Probability', ascending=False).head(10))

# Save at-risk customer list (optional)
at_risk_customers.to_csv('At_Risk_Customers.csv', index=False)
print("\n✅ At-risk customer list saved as 'At_Risk_Customers.csv'!")
