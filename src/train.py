from preprocessing import load_data, preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


X, y = load_data("data/heart.csv")

X_train, X_test, y_train, y_test = preprocess_data(X, y)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

joblib.dump(model, "models/heart_model.pkl")