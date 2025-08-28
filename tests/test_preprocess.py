import unittest
import pandas as pd
from scripts.preprocess import load_data, preprocess_data

class TestPreprocess(unittest.TestCase):
    def test_load_data(self):
        # Load a small sample dataset
        df = load_data("data/heart_dataset.csv")
        self.assertIsInstance(df, pd.DataFrame)  # check it's a DataFrame

    def test_preprocess_data(self):
        df = pd.DataFrame({
            "age": [29, 50, None],
            "cholesterol": [200, None, 180]
        })
        processed = preprocess_data(df)
        # After preprocessing, there should be no missing values
        self.assertFalse(processed.isnull().values.any())

if __name__ == "__main__":
    unittest.main()
