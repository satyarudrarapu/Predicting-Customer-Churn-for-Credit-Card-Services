from google.colab import files
import pandas as pd
# Upload the file
uploaded = files.upload()
# Get the filename
filename = list(uploaded.keys())[0]
# Read the dataset
df = pd.read_csv(filename)
# Display the first few rows
df.head()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from google.colab import files
# Upload the dataset in Google Colab
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
#Transaction Frequency & Recency (Proxy Features)
df['zero_balance'] = (df['balance'] == 0).astype(int)  
#Declining Spending Patterns (Proxy)
df['low_credit_score'] = (df['credit_score'] < df['credit_score'].median()).astype(int)
#Late or Missed Payments (Proxy)
df['high_risk'] = ((df['credit_score'] < df['credit_score'].quantile(0.25)) & 
                   (df['balance'] > df['balance'].median())).astype(int)
#Customer Complaints or Service Interactions (Placeholder)
df['customer_complaints'] = np.random.randint(0, 2, size=len(df))  
#Normalize Numerical Features
scaler = StandardScaler()
num_cols = ['credit_score', 'age', 'balance', 'estimated_salary']
df[num_cols] = scaler.fit_transform(df[num_cols])
#One-Hot Encode Categorical Features (FIXED)
encoder = OneHotEncoder(drop='first', sparse_output=False)  # FIXED HERE
encoded_cols = encoder.fit_transform(df[['country', 'gender']])
# Convert to DataFrame
encoded_feature_names = encoder.get_feature_names_out(['country', 'gender'])
df_encoded = pd.DataFrame(encoded_cols, columns=encoded_feature_names)
# Merge encoded data
df = pd.concat([df, df_encoded], axis=1)
# Drop original categorical columns
df.drop(columns=['country', 'gender', 'tenure'], inplace=True, errors='ignore')
# Display dataset structure
print(df.info())
#Save and Download Processed Data
df.to_csv("Processed_Customer_Churn.csv", index=False)
files.download("Processed_Customer_Churn.csv")
