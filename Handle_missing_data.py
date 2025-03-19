import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
file_path = 'Cleaned_Bank_Customer_Churn.csv'  # Ensure the file is in the same directory
data = pd.read_csv(file_path)
# Set a consistent theme for plots
sns.set_theme(style="whitegrid")
# Step 1: Display basic dataset information
print("Dataset Overview:")
print(data.info())
print("\nMissing Values Per Column:")
print(data.isnull().sum())
# Step 2: Identify negative or unrealistic values
negative_credit_scores = data[data['credit_score'] < 0]
print("\nNegative Credit Scores:")
print(negative_credit_scores)
unrealistic_ages = data[(data['age'] < 0) | (data['age'] > 100)]
print("\nUnrealistic Ages:")
print(unrealistic_ages)
# Step 3: Handle missing and erroneous data
# Assuming no missing values from earlier check
# Replace negative credit scores with the column's median
if not negative_credit_scores.empty:
    median_credit_score = data['credit_score'][data['credit_score'] >= 0].median()
    data.loc[data['credit_score'] < 0, 'credit_score'] = median_credit_score
    print("\nNegative credit scores replaced with median:", median_credit_score)
# Replace unrealistic ages with the column's median
if not unrealistic_ages.empty:
    median_age = data['age'][(data['age'] >= 0) & (data['age'] <= 100)].median()
    data.loc[(data['age'] < 0) | (data['age'] > 100), 'age'] = median_age
    print("\nUnrealistic ages replaced with median:", median_age)
# Check for missing values after handling erroneous data
print("\nPost-Cleaning Missing Values:")
print(data.isnull().sum())
# Step 4: Perform EDA (Exploratory Data Analysis)
# Distribution of credit scores
plt.figure(figsize=(8, 5))
sns.histplot(data['credit_score'], bins=30, kde=True, color="blue")
plt
