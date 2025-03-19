# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
# Predict on validation set
y_val_pred_dt = dt_model.predict(X_val)
y_val_prob_dt = dt_model.predict_proba(X_val)[:, 1]
# Evaluate Decision Tree
dt_accuracy = accuracy_score(y_val, y_val_pred_dt)
dt_precision = precision_score(y_val, y_val_pred_dt)
dt_recall = recall_score(y_val, y_val_pred_dt)
dt_auc_roc = roc_auc_score(y_val, y_val_prob_dt)
print(f"\n✅ Decision Tree Performance:")
print(f"Accuracy: {dt_accuracy:.2f}, Precision: {dt_precision:.2f}, Recall: {dt_recall:.2f}, AUC-ROC: {dt_auc_roc:.2f}")

