# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
# Predict on validation set
y_val_pred_xgb = xgb_model.predict(X_val)
y_val_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
# Evaluate XGBoost
xgb_accuracy = accuracy_score(y_val, y_val_pred_xgb)
xgb_precision = precision_score(y_val, y_val_pred_xgb)
xgb_recall = recall_score(y_val, y_val_pred_xgb)
xgb_auc_roc = roc_auc_score(y_val, y_val_prob_xgb)
print(f"\n✅ XGBoost Performance:")
print(f"Accuracy: {xgb_accuracy:.2f}, Precision: {xgb_precision:.2f}, Recall: {xgb_recall:.2f}, AUC-ROC: {xgb_auc_roc:.2f}")
