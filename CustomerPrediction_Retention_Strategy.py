import pandas as pd
import numpy as np
from google.colab import files
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Upload dataset (Google Colab users)
uploaded = files.upload()

# Load the dataset
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)

# Display dataset structure
print("\n📊 First 5 rows of the dataset:")
print(data.head())

# Check available columns
print("\n✅ Available columns in dataset:", list(data.columns))

# Handle missing columns
if 'gender' in data.columns:
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
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
y = data['churn']  # Assuming target column is 'churn'

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model for churn prediction
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict churn probabilities
y_test_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Segment high-value customers (High balance, multiple products, high salary)
high_value_customers = data[
    (data['balance'] > data['balance'].quantile(0.75)) &
    (data['products_number'] >= 2) &
    (data['estimated_salary'] > data['estimated_salary'].quantile(0.75))
]

# Segment at-risk customers (High churn probability)
test_results = X_test.copy()
test_results['Churn_Probability'] = y_test_prob
test_results['Actual_Churn'] = y_test.values
at_risk_customers = test_results[test_results['Churn_Probability'] > 0.5]

# Provide retention strategies
print("\n🎯 Retention Strategies:\n")

# 1️⃣ Incentives for High-Value Customers
print("💰 Offer exclusive rewards for high-value customers:")
print("- Personalized cashback or loyalty points")
print("- Special discounts on premium services")
print("- Dedicated customer support for VIP clients\n")

# 2️⃣ Targeted Communication Campaigns for At-Risk Customers
print("📩 Send proactive messages to at-risk customers:")
print("- Personalized emails with special retention offers")
print("- Discounts or perks for continued engagement")
print("- Customer service calls to address concerns\n")

# Save segmented customer data (optional)
high_value_customers.to_csv('High_Value_Customers.csv', index=False)
at_risk_customers.to_csv('At_Risk_Customers.csv', index=False)

print("\n✅ High-value customer list saved as 'High_Value_Customers.csv'!")
print("✅ At-risk customer list saved as 'At_Risk_Customers.csv'!")

