# Train LightGBM
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)
# Predict on validation set
y_val_pred_lgbm = lgbm_model.predict(X_val)
y_val_prob_lgbm = lgbm_model.predict_proba(X_val)[:, 1]
# Evaluate LightGBM
lgbm_accuracy = accuracy_score(y_val, y_val_pred_lgbm)
lgbm_precision = precision_score(y_val, y_val_pred_lgbm)
lgbm_recall = recall_score(y_val, y_val_pred_lgbm)
lgbm_auc_roc = roc_auc_score(y_val, y_val_prob_lgbm)
print(f"\n✅ LightGBM Performance:")
print(f"Accuracy: {lgbm_accuracy:.2f}, Precision: {lgbm_precision:.2f}, Recall: {lgbm_recall:.2f}, AUC-ROC: {lgbm_auc_roc:.2f}")
