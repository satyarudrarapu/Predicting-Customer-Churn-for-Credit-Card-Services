nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_smote.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_smote, y_train_smote, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=0)
y_test_prob_nn = nn_model.predict(X_test).flatten()
