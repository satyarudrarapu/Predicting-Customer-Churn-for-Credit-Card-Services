# Compute AUC-ROC scores
stack_auc = roc_auc_score(y_test, y_test_prob_stack)
nn_auc = roc_auc_score(y_test, y_test_prob_nn)
print(f"✅ Stacking AUC-ROC: {stack_auc:.2f}")
print(f"✅ Neural Network AUC-ROC: {nn_auc:.2f}")
