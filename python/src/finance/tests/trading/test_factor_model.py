import unittest
import pandas as pd
from finance.trading.factor.preprocess import *
from finance.trading.factor.benchmark import *
from finance.trading.factor.proj_grad import *
from finance.constant import BASE_PATH


class TestFactorModel(unittest.TestCase):
    def setUp(self) -> None:
        self.base_path = BASE_PATH.parent.parent.parent.parent
        self.max_lag = 250
        self.n_factors = 10
        path_price = f"{self.base_path}/data/prices.csv"
        self.prices = pd.read_csv(path_price, sep=";")
        preprocessor = PreProcess(self.max_lag)
        self.X_train0, self.Y_train0 = preprocessor.split_x_y(
            self.prices, DataType.PRICE
        )
        path_ret = f"{self.base_path}/data/returns.csv"
        self.data = pd.read_csv(path_ret, sep=";")
        self.X_train1, self.Y_train1 = preprocessor.split_x_y(
            self.data, DataType.RETURN
        )

    def test_preprocess(self) -> None:
        length_per_asset = (
            self.prices.shape[0] - 1 - self.max_lag
        )  # The -1 is due to the dropna in pct_change
        self.assertEqual(self.X_train0.shape[1], self.max_lag)
        self.assertEqual(
            self.X_train0.shape[0],
            length_per_asset * self.prices.shape[1],
        )
        self.assertEqual(self.Y_train0.shape[1], length_per_asset)

    def test_benchmark(self) -> None:
        benchmark = Benchmark(
            lag=self.max_lag, n_factors=self.n_factors, n_iter=10, eps=1e-6
        )
        stiefel, beta = benchmark.train(self.X_train1, self.Y_train1)
        self.assertEqual(stiefel.shape[0], self.max_lag)
        self.assertEqual(stiefel.shape[1], self.n_factors)
        self.assertEqual(beta.shape[0], self.n_factors)

    def test_proj_grad(self) -> None:
        proj = ProjGrad(
            lag=self.max_lag,
            n_factors=self.n_factors,
            eps=1.0e-6,
            step_stiefel=0.1,
            step_beta=0.05,
            random_state=1234,
            n_iter=100,
            use_benchmark=False,
        )
        benchmark = Benchmark(
            lag=self.max_lag, n_factors=self.n_factors, n_iter=10, eps=1e-6
        )
        stiefel0, beta0 = benchmark.train(self.X_train1, self.Y_train1)
        stiefel, beta = proj.train(self.X_train1, self.Y_train1, stiefel0, beta0)
        metric_proj_grad = proj.metric_train(
            stiefel, beta, self.X_train1, self.Y_train1
        )
        metric_benchmark = benchmark.metric_train(
            stiefel0, beta0, self.X_train1, self.Y_train1
        )
        self.assertGreaterEqual(metric_proj_grad, metric_benchmark)
