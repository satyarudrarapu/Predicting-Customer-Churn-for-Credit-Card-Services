stack_model = StackingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
], final_estimator=LogisticRegression())
stack_model.fit(X_train_smote, y_train_smote)
y_test_prob_stack = stack_model.predict_proba(X_test)[:, 1]
