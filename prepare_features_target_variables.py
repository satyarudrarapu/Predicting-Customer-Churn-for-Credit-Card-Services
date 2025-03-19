# Separate features and target variable
X = df.drop(columns=['churn'])  # Features
y = df['churn']  # Target variable (0 = No Churn, 1 = Churn)

# Ensure no missing values
X.fillna(0, inplace=True)
