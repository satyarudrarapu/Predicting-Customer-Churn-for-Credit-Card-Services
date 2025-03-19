from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
# Example data (true labels and predicted probabilities or predictions)
# Replace these with your actual data
y_true = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  # Ground truth labels
y_pred = [0, 1, 1, 0, 0, 0, 1, 0, 1, 1]  # Predicted labels
y_prob = [0.1, 0.9, 0.8, 0.2, 0.4, 0.1, 0.85, 0.05, 0.6, 0.9]  # Predicted probabilities for positive class
# Calculate evaluation metrics
auc_roc = roc_auc_score(y_true, y_prob)  # AUC-ROC using predicted probabilities
precision = precision_score(y_true, y_pred)  # Precision using predicted labels
recall = recall_score(y_true, y_pred)  # Recall using predicted labels
f1 = f1_score(y_true, y_pred)  # F1-Score using predicted labels
# Display results
print(f"AUC-ROC: {auc_roc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
AUC-ROC: 0.96
Precision: 0.80
Recall: 0.80
F1-Score: 0.80
