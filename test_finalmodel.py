After tuning the model (if needed), evaluate it on the test set.
# Predict on the test set
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]
# Compute final test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_auc_roc = roc_auc_score(y_test, y_test_prob)
# Display final test performance
print(f"\nFinal Test Metrics:")
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")
print(f"AUC-ROC: {test_auc_roc:.2f}")
