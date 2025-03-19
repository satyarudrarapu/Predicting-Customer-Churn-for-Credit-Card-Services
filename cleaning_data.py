import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load the dataset
data = pd.read_csv("Bank Customer Churn Prediction.csv")
# 1. Drop unnecessary columns
# Assuming 'customer_id' is not needed for analysis
data.drop(columns=['customer_id'], inplace=True)
# 2. Handle categorical variables
# Encode 'country' and 'gender'
label_encoder = LabelEncoder()
data['country'] = label_encoder.fit_transform(data['country'])
data['gender'] = label_encoder.fit_transform(data['gender'])
# 3. Check for duplicates and remove them
data.drop_duplicates(inplace=True)
# 4. Scaling numerical columns
# Columns to scale: 'credit_score', 'age', 'balance', 'estimated_salary'
scaler = StandardScaler()
numerical_columns = ['credit_score', 'age', 'balance', 'estimated_salary']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
# 5. Final clean dataset summary
print("Data shape after cleaning:", data.shape)
print("Sample data:")
print(data.head())
# Save the cleaned data to a new CSV file
data.to_csv("Cleaned_Bank_Customer_Churn.csv", index=False)


