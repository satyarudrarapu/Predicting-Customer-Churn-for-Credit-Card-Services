import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
# Upload dataset (if needed)
from google.colab import files
uploaded = files.upload()
# Load processed data
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
# Display first few rows
df.head()
