# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
# Print detailed classification report
print("\n✅ Classification Report:\n", classification_report(y_test, y_test_pred))
