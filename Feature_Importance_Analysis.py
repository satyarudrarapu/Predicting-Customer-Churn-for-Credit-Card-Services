# Extract feature importance (only for tree-based models like Random Forest, XGBoost, LightGBM)
feature_importance = best_rf_model.feature_importances_
# Convert to DataFrame
feat_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
# Sort by importance
feat_importance_df = feat_importance_df.sort_values(by="Importance", ascending=False)
# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_importance_df, palette="viridis")
plt.title("Feature Importance for Churn Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.show()
# Print Top Features
print("\n✅ Top 10 Features Driving Churn:")
print(feat_importance_df.head(10))
