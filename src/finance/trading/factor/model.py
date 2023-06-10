from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm
import pandas as pd
from typing import Tuple, Optional
from scipy.linalg import svd, inv, sqrtm
from numba import njit
from tqdm import tqdm


@dataclass
class Model:
    lag: int
    n_factors: int
    eps: float = 1e-6

    @property
    def name(self):
        return f"{self.__class__.__name__} model"

    # The following methods are taken and adapted from QRT Challenges
    def random_stiefel(self) -> np.ndarray:
        gaussian = np.random.randn(self.lag, self.n_factors)
        # Apply Gram-Schmidt algorithm to the columns of the matrix gaussian
        stiefel = np.linalg.qr(gaussian)[0]
        return stiefel

    def check_orthonormality(self, stiefel: np.ndarray) -> bool:
        n_factors = stiefel.shape[1]
        error = pd.DataFrame(stiefel.T @ stiefel - np.eye(n_factors)).abs()
        return not any(error.unstack() > self.eps)

    def metric_train(
        self,
        stiefel: np.ndarray,
        beta: np.ndarray,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
    ) -> float:
        if not self.check_orthonormality(stiefel):
            return -1.0

        Y_pred: pd.DataFrame = X_train @ stiefel @ beta
        Y_pred = Y_pred.unstack().T

        Y_true = Y_train.div(np.sqrt((Y_train**2).sum()), 1).reset_index(drop=True)
        Y_pred = Y_pred.div(np.sqrt((Y_pred**2).sum()), 1).reset_index(drop=True)

        mean_overlap = (Y_true * Y_pred).sum().mean()

        return mean_overlap


@dataclass
class BenchMark(Model):
    n_iter: int = 1000
    random_state: int = 1234

    def __post_init__(self):
        self.max_metric = -1.0

    @staticmethod
    def fit_beta(
        stiefel: np.ndarray, X_train: pd.DataFrame, Y_train: pd.DataFrame
    ) -> np.ndarray:
        predictors: pd.DataFrame = X_train @ stiefel
        targets: pd.DataFrame = Y_train.T.stack()
        beta: pd.DataFrame = (
            np.linalg.inv(predictors.T @ predictors) @ predictors.T @ targets
        )
        return beta.to_numpy()

    def train(
        self, X_train: pd.DataFrame, Y_train: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.random_state)
        stiefel_star = np.empty((self.lag, self.n_factors))
        beta_star = np.empty((self.n_factors,))
        iterator = range(self.n_iter)
        for iteration in tqdm(iterator, total=self.n_iter, desc=self.name):
            # Generate a uniform random Stiefel matrix and fit beta
            # with minimal mean square prediction error on the training data set
            stiefel = self.random_stiefel()
            beta = self.fit_beta(stiefel, X_train, Y_train)
            # Compute the metric on the training set and keep the best result
            m = self.metric_train(stiefel, beta, X_train, Y_train)
            if m > self.max_metric:
                tqdm.write(f"Iteration {iteration} with best train metric: {m}")
                self.max_metric = m
                stiefel_star = stiefel
                beta_star = beta
        return stiefel_star, beta_star


@dataclass
class ProjGrad(Model):
    step_stiefel: float = 0.1
    step_beta: float = 0.05
    random_state: int = 1234
    n_iter: int = 1000
    use_benchmark: bool = False

    def __post_init__(self):
        if self.use_benchmark:
            self.__benchmark = BenchMark(
                n_iter=self.n_iter, random_state=self.random_state
            )

    @staticmethod
    def project(stiefel: np.ndarray) -> np.ndarray:
        u, _, v = svd(stiefel, full_matrices=False)
        return u @ v

    @staticmethod
    # @njit
    def grad_stiefel(
        returns: np.ndarray,
        beta: np.ndarray,
        factors: np.ndarray,
        stiefel: np.ndarray,
        prediction: np.ndarray,
        lagged_returns: np.ndarray,
        identity_lag: np.ndarray,
    ):
        return list(
            map(
                lambda factor: (
                    returns
                    @ (
                        beta[factor] * identity_lag
                        - beta[factor]
                        * stiefel
                        @ beta.reshape(-1, 1)
                        @ prediction.reshape(1, -1)
                        @ lagged_returns
                        / np.sqrt(np.linalg.norm(prediction, 2)) ** 3
                    )
                ).reshape(-1, 1),
                factors,
            )
        )

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
    def get_indices_from(index: int, n_assets: int):
        return list(range(index, index + n_assets))

    def train(
        self,
        stiefel0: Optional[np.ndarray],
        beta0: Optional[np.ndarray],
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
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
        identity_lag = np.eye(self.lag, self.lag)
        identity_factors = np.eye(self.n_factors, self.n_factors)
        factors = np.arange(0, self.n_factors)
        indices = X_train.index
        dates = indices.get_level_values(0).unique()
        n_dates = len(dates)
        assets = indices.get_level_values(1).unique()
        n_assets = len(assets)
        for index in tqdm(range(n_dates), total=n_dates, desc=self.name):
            date = index + self.lag
            return_date = Y_train.loc[:, date].to_numpy()
            # print(f"return_date shape: {return_date.shape}")
            # lagged_returns = X_train.loc[self.get_index(date, assets), :].to_numpy()
            lagged_returns = X_train.iloc[
                self.get_indices_from(index, n_assets), :
            ].to_numpy()
            # print(f"lagged_returns shape: {lagged_returns.shape}")
            prediction = lagged_returns @ stiefel @ beta.reshape(-1, 1)
            # print(f"prediction shape: {prediction.shape}")
            # print(f"prediction.T shape: {prediction.T.shape}")
            returns = return_date.T @ lagged_returns / np.sqrt(norm(return_date, 2))
            # print(f"returns shape: {returns.shape}")

            # gradients
            grad_stiefel = np.hstack(
                self.grad_stiefel(
                    returns,
                    beta,
                    factors,
                    stiefel,
                    prediction,
                    lagged_returns,
                    identity_lag,
                )
            )
            grad_beta = self.grad_beta(
                returns,
                beta,
                stiefel,
                prediction,
                lagged_returns,
                identity_factors,
            )

            # updates
            beta = beta + self.step_beta * grad_beta
            # beta = fitBeta(A).reshape(-1,1)
            # beta = fitBeta(A)
            stiefel_temp = stiefel + self.step_stiefel * grad_stiefel
            # stiefel = self.project(stiefel_temp)
            stiefel = stiefel_temp @ inv(sqrtm(stiefel_temp.T @ stiefel_temp))
            # print(f"beta shape: {beta.shape}")

            m = self.metric_train(stiefel, beta, X_train, Y_train)

            # tqdm.write(f"metric {m}")
            if m > max_metric:
                tqdm.write(f"Date: {date} with best train metric: {m}")
                max_metric = m
                stiefel_star = stiefel
                beta_star = beta

        return stiefel_star, beta_star
