import unittest
import pandas as pd
from finance.trading.factor.preprocess import PreProcess
from finance.trading.factor.model import BenchMark


class TestFactorModel(unittest.TestCase):
    def setUp(self) -> None:
        path = "/home/georges/compet/orycterope/prices.csv"
        self.prices = pd.read_csv(path, sep=";")
        self.max_lag = 100
        self.n_factors = 10
        preprocessor = PreProcess(self.max_lag)
        self.X_train, self.Y_train = preprocessor.split_x_y(self.prices)

    def test_preprocess(self) -> None:
        length_per_asset = (
            self.prices.shape[0] - 1 - self.max_lag
        )  # The -1 is due to the dropna in pct_change
        self.assertEqual(self.X_train.shape[1], self.max_lag)
        self.assertEqual(
            self.X_train.shape[0],
            length_per_asset * self.prices.shape[1],
        )
        self.assertEqual(self.Y_train.shape[1], length_per_asset)

    def test_benchmark(self) -> None:
        benchmark = BenchMark(lag=self.max_lag, n_factors=self.n_factors, n_iter=3)
        stiefel, beta = benchmark.train(self.X_train, self.Y_train)
        self.assertEqual(stiefel.shape[0], self.max_lag)
        self.assertEqual(stiefel.shape[1], self.n_factors)
        self.assertEqual(beta.shape[0], self.n_factors)
