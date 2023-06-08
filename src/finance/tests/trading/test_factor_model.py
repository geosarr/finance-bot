import unittest
import pandas as pd
from finance.trading.factor_model.preprocess import *


class TestLoader(unittest.TestCase):
    def setUp(self) -> None:
        path = "/home/georges/compet/orycterope/prices.csv"
        self.prices = pd.read_csv(path, sep=";")
        self.max_lag = 100
        self.preprocessor = PreProcess(self.max_lag)

    def test_preprocess(self) -> None:
        X_train, Y_train = self.preprocessor.split_x_y(self.prices)
        length_per_asset = (
            self.prices.shape[0] - 1 - self.max_lag
        )  # The -1 is due to the dropna in pct_change
        self.assertEqual(X_train.shape[1], self.max_lag)
        self.assertEqual(
            X_train.shape[0],
            length_per_asset * self.prices.shape[1],
        )
        self.assertEqual(Y_train.shape[1], length_per_asset)
