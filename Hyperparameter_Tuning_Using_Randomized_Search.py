from scipy.stats import randint
# Define parameter grid
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
}
# Initialize Random Search
rf_random = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                               param_distributions=param_dist,
                               n_iter=10, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
# Fit on training data
rf_random.fit(X_train, y_train)
# Best parameters
print("\n✅ Best Random Forest Parameters:", rf_random.best_params_)
# Predict on validation set with best model
best_rf_model = rf_random.best_estimator_
y_val_pred_best_rf = best_rf_model.predict(X_val)
y_val_prob_best_rf = best_rf_model.predict_proba(X_val)[:, 1]

# Evaluate best Random Forest model
best_rf_auc_roc = roc_auc_score(y_val, y_val_prob_best_rf)
print(f"✅ Best Random Forest AUC-ROC: {best_rf_auc_roc:.2f}")
