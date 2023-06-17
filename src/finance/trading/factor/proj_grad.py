from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm
import pandas as pd
from typing import Tuple, Optional
from scipy.linalg import svd, inv, sqrtm
from numba import guvectorize, float64, int64
from tqdm.auto import tqdm
from finance.trading.factor.__model import Model
from finance.trading.factor.benchmark import Benchmark


@dataclass
class ProjGrad(Model):
    step_stiefel: float = 0.1
    step_beta: float = 0.05
    random_state: int = 1234
    n_iter: int = 1000
    use_benchmark: bool = False

    def __post_init__(self):
        if self.use_benchmark:
            self.__benchmark = Benchmark(
                lag=self.lag,
                n_factors=self.n_factors,
                eps=self.eps,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        stiefel0: Optional[np.ndarray] = None,
        beta0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_benchmark:
            stiefel0, beta0 = self.__benchmark.train(X_train, Y_train)
        elif stiefel0 is None or beta0 is None:
            raise ValueError(
                "Initial values should be provided for stiefel0 and beta0."
            )

        stiefel = stiefel0.copy()
        beta = beta0.copy()
        stiefel_star = stiefel.copy()
        beta_star = beta.copy()
        if len(beta.shape) == 1:
            beta = beta.reshape(-1, 1)
        max_metric = self.metric_train(stiefel, beta, X_train, Y_train)
        identity_lag = np.eye(self.lag)
        identity_factors = np.eye(self.n_factors)
        factors = np.arange(self.n_factors)
        n_assets = Y_train.shape[0]
        n_dates = X_train.shape[0] // n_assets
        gen = np.random.default_rng(self.random_state)
        iterations = gen.choice(np.arange(n_dates), self.n_iter, replace=True)
        for index in tqdm(iterations, desc=self.name):
            return_date = Y_train[:, index]
            # print(f"return_date shape: {return_date.shape}")
            # print(f"\n{date} return_date: \n{return_date}")
            # lagged_returns = X_train.loc[self.get_index(date, assets), :].to_numpy()
            start = index * n_assets
            end = (index + 1) * n_assets
            lagged_returns = X_train[start:end, :]
            # print(f"lagged_returns shape: {lagged_returns.shape}")
            # print(f"\n{date} lagged_returns: \n{lagged_returns}")
            prediction = lagged_returns @ stiefel @ beta.reshape(-1, 1)
            # print(f"prediction shape: {prediction.shape}")
            # print(f"prediction.T shape: {prediction.T.shape}")
            # print(f"\n{date} prediction: \n{prediction}")
            returns = return_date.T @ lagged_returns / np.sqrt(norm(return_date, 2))
            # print(f"returns shape: {returns.shape}")
            # print(f"\n{date} returns: \n{returns}")

            # gradients
            grad_stiefel = self.grad_stiefel(
                returns,
                beta.reshape(-1, 1),
                factors,
                stiefel,
                prediction.reshape(1, -1),
                lagged_returns,
                identity_lag,
            ).T.reshape(self.lag, self.n_factors)
            # print(f"\n{date} grad_stiefel: \n{grad_stiefel}")
            grad_beta = self.grad_beta(
                returns,
                beta,
                stiefel,
                prediction,
                lagged_returns,
                identity_factors,
            )
            # print(f"\n{date} grad_beta: \n{grad_beta}")

            # updates
            beta = beta + self.step_beta * grad_beta
            # beta = fitBeta(A)
            # print(f"\n{date} beta: \n{beta}")
            stiefel_temp = stiefel + self.step_stiefel * grad_stiefel
            stiefel = self.project(stiefel_temp)
            # stiefel = stiefel_temp @ inv(sqrtm(stiefel_temp.T @ stiefel_temp))
            # print(f"\n{date} stiefel: \n{stiefel}")

            m = self.metric_train(stiefel, beta, X_train, Y_train)

            if m > max_metric:
                tqdm.write(f"Iteration: {index+1} with best train metric: {m}")
                max_metric = m
                stiefel_star = stiefel
                beta_star = beta

        return stiefel_star, beta_star

    @staticmethod
    @guvectorize(
        [
            (
                float64[:],
                float64[:, :],
                int64,
                float64[:, :],
                float64[:, :],
                float64[:, :],
                float64[:, :],
                float64[:, :],
            )
        ],
        "(q),(m,k),(),(p,m),(k,n),(n,q),(q,q)->(k,q)",
        nopython=True,
    )
    def grad_stiefel(
        returns: np.ndarray,
        beta: np.ndarray,
        factor: int,
        stiefel: np.ndarray,
        prediction: np.ndarray,
        lagged_returns: np.ndarray,
        identity_lag: np.ndarray,
        res: np.ndarray,
    ):
        res[:, :] = (
            returns
            @ (
                beta[factor][0] * identity_lag
                - beta[factor][0]
                * stiefel
                @ beta
                @ prediction
                @ lagged_returns
                / np.sqrt(np.linalg.norm(prediction, 2)) ** 3
            )
        ).reshape(1, -1)[:, :]

    @staticmethod
    def grad_beta(
        returns: np.ndarray,
        beta: np.ndarray,
        stiefel: np.ndarray,
        prediction: np.ndarray,
        lagged_returns: np.ndarray,
        identity_factors: np.ndarray,
    ):
        return (
            returns
            @ stiefel
            @ (
                identity_factors
                - beta.reshape(-1, 1)
                @ prediction.reshape(1, -1)
                @ lagged_returns
                @ stiefel
                / np.sqrt(np.linalg.norm(prediction, 2)) ** 3
            )
        ).reshape(-1, 1)

    @staticmethod
    def project(stiefel: np.ndarray) -> np.ndarray:
        u, _, v = svd(stiefel, full_matrices=False)
        return u @ v
