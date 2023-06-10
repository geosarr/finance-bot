import unittest
import pandas as pd
from finance.trading.factor.preprocess import PreProcess
from finance.trading.factor.model import *


class TestFactorModel(unittest.TestCase):
    def setUp(self) -> None:
        path = "/home/georges/compet/orycterope/prices.csv"
        self.prices = pd.read_csv(path, sep=";")
        self.max_lag = 50
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
        benchmark = BenchMark(
            lag=self.max_lag, n_factors=self.n_factors, n_iter=3, eps=1e-6
        )
        stiefel, beta = benchmark.train(self.X_train, self.Y_train)
        self.assertEqual(stiefel.shape[0], self.max_lag)
        self.assertEqual(stiefel.shape[1], self.n_factors)
        self.assertEqual(beta.shape[0], self.n_factors)

    def test_proj_grad(self):
        proj = ProjGrad(
            lag=self.max_lag,
            n_factors=self.n_factors,
            eps=1.0e-6,
            step_stiefel=0.2,
            step_beta=0.05,
            random_state=123,
            n_iter=10,
        )
        beta0 = np.ones(self.n_factors)
        stiefel0 = np.ones((self.max_lag, self.n_factors))
        stiefel, beta = proj.train(stiefel0, beta0, self.X_train, self.Y_train)
        print(stiefel, beta)
