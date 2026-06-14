import joblib
import pandas as pd

from preprocessing import load_data, preprocess_data


X, y = load_data("data/heart.csv")

X_train, X_test, y_train, y_test = preprocess_data(X, y)

model = joblib.load("models/heart_model.pkl")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Predictions:")
print(predictions[:10])

print("\nProbabilities:")
print(probabilities[:10])