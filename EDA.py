import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
file_path = '/mnt/data/Cleaned_Bank_Customer_Churn.csv'
# Set a consistent theme for plots
sns.set_theme(style="whitegrid")
# 1. Understand Target Distribution
churn_counts = data['churn'].value_counts(normalize=True)
plt.figure(figsize=(6, 4))
sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="viridis")
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Proportion")
plt.xticks([0, 1], ["No Churn","Churn"])
plt.show()
# 2. Explore Numerical Features
numerical_features = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=data, x=feature, hue="churn", kde=True, palette="viridis", bins=30)
    plt.title(f"Distribution of {feature} by Churn")
    plt.show()
# 3. Analyze Categorical Features
categorical_features = ['country', 'gender', 'products_number', 'credit_card', 'active_member']
for feature in categorical_features:
    plt.figure(figsize=(6, 4))
    churn_counts_by_category = data.groupby(feature)['churn'].mean()
    sns.barplot(x=churn_counts_by_category.index, y=churn_counts_by_category.values, palette="viridis")
    plt.title(f"Churn Rate by {feature}")
    plt.xlabel(feature.capitalize())
    plt.ylabel("Churn Rate")
    plt.show()
# 4. Correlation Heatmap
plt.figure(figsize=(10, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix")
plt.show()
