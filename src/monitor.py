import logging
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score
import joblib

from preprocessing import load_data, preprocess_data


logging.basicConfig(
    filename="monitor.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

X, y = load_data("data/heart.csv")

X_train, X_test, y_train, y_test = preprocess_data(X, y)

model = joblib.load("models/heart_model.pkl")

pred_original = model.predict(X_test)

original_acc = balanced_accuracy_score(y_test, pred_original)

logging.info(f"Original Balanced Accuracy: {original_acc}")

X_drifted = pd.DataFrame(X_test.copy())

X_drifted[0] = X_drifted[0] + 30

pred_drifted = model.predict(X_drifted)

drifted_acc = balanced_accuracy_score(y_test, pred_drifted)

logging.info(f"Drifted Balanced Accuracy: {drifted_acc}")

print("Original Balanced Accuracy:", original_acc)
print("Drifted Balanced Accuracy:", drifted_acc)

print("\nKS Test Results:")

for col in range(X_test.shape[1]):
    statistic, p_value = ks_2samp(X_test[:, col], X_drifted[col])

    print(f"Feature {col}: p-value = {p_value}")

    if p_value < 0.05:
        print(f"-> Drift detected in Feature {col}")

accuracy_values = [original_acc, drifted_acc]

plt.plot(
    ["Original", "Drifted"],
    accuracy_values,
    marker='o'
)

plt.title("Balanced Accuracy Drift Comparison")
plt.ylabel("Balanced Accuracy")

plt.show()