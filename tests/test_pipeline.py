import unittest
import numpy as np
import joblib

from src.preprocessing import load_data, preprocess_data


class TestCardioCarePipeline(unittest.TestCase):

    def setUp(self):
        X, y = load_data("data/heart.csv")
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(X, y)
        self.model = joblib.load("models/heart_model.pkl")

    def test_prediction_shape(self):
        predictions = self.model.predict(self.X_test)
        self.assertEqual(predictions.shape[0], self.X_test.shape[0])

    def test_probability_range_and_sum(self):
        probabilities = self.model.predict_proba(self.X_test)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_chol_range_validation(self):
        X, y = load_data("data/heart.csv")
        self.assertTrue((X["chol"] >= 0).all())
        self.assertTrue((X["chol"] <= 600).all())

    def test_deterministic_pipeline(self):
        X, y = load_data("data/heart.csv")
        result1 = preprocess_data(X, y)
        result2 = preprocess_data(X, y)

        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])


if __name__ == "__main__":
    unittest.main()