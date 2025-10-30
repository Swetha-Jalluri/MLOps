import unittest
import numpy as np
import pandas as pd
from src.wine_model import preprocess_data, train_model, evaluate_model

class TestWineModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = pd.DataFrame(rng.normal(size=(40, 11)))
        self.y = pd.Series(rng.integers(0, 2, size=40))
        self.X_train, self.X_test = self.X.iloc[:32], self.X.iloc[32:]
        self.y_train, self.y_test = self.y.iloc[:32], self.y.iloc[32:]

    def test_end_to_end(self):
        X_trs, X_tes, scaler = preprocess_data(self.X_train, self.X_test)
        model = train_model(X_trs, self.y_train, n_estimators=5)
        metrics = evaluate_model(model, X_tes, self.y_test)
        self.assertIn("accuracy", metrics)

if __name__ == "__main__":
    unittest.main()
