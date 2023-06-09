from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.linalg import svd  # , inv, sqrtm
from numba import njit


@dataclass
class Model:
    lag: int = 250
    n_factors: int = 10
    eps: float = 1e-6

    # The following methods are taken and adapted from QRT Challenges
    def random_stiefel(self) -> np.ndarray:
        gaussian = np.random.randn(self.lag, self.n_factors)
        # Apply Gram-Schmidt algorithm to the columns of the matrix gaussian
        stiefel = np.linalg.qr(gaussian)[0]
        return stiefel

    def check_orthonormality(self, stiefel: np.ndarray) -> bool:
        n_factors = stiefel.shape[1]
        error = pd.DataFrame(stiefel.T @ stiefel - np.eye(n_factors)).abs()
        return any(error.unstack() > self.eps)

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

        Ytrue = Y_train.div(np.sqrt((Y_train**2).sum()), 1)
        Ypred = Y_pred.div(np.sqrt((Y_pred**2).sum()), 1)

        mean_overlap = (Ytrue * Ypred).sum().mean()

        return mean_overlap


@dataclass
class BenchMark(Model):
    n_iter: int = 1000
    random_state: int = 1234

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
        max_metric = -1
        stiefel_star, beta_star = np.empty((self.lag, self.n_factors)), np.empty(
            (self.n_factors,)
        )
        for iteration in range(self.n_iter):
            # Generate a uniform random Stiefel matrix and fit beta
            # with minimal mean square prediction error on the training data set
            stiefel = self.random_stiefel()
            beta = self.fit_beta(stiefel, X_train, Y_train)
            # Compute the metric on the training set and keep the best result
            m = self.metric_train(stiefel, beta, X_train, Y_train)
            if m > max_metric:
                print(iteration, "metric_train:", m)
                max_metric = m
                stiefel_star = stiefel
                beta_star = beta
        return stiefel_star, beta_star


@dataclass
class ProjGrad(Model):
    n_iter: int = 1000
    step_stiefel: float
    step_beta: float
    random_state: int

    @staticmethod
    def project(stiefel: np.ndarray) -> np.ndarray:
        u, _, v = svd(stiefel, full_matrices=False)
        return u @ v

    @njit
    @staticmethod
    def grad_stiefel(
        returns: np.ndarray,
        beta: np.ndarray,
        factor: int,
        stiefel: np.ndarray,
        prediction: np.ndarray,
        lagged_returns: np.ndarray,
        identity_lag: np.ndarray,
    ):
        return (
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
        ).reshape(-1, 1)

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

    def train(
        self,
        stiefel0: np.ndarray,
        beta0: np.ndarray,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(beta0.shape) == 1:
            beta = beta0.reshape(-1, 1)
        max_metric = self.metric_train(stiefel0, beta)
        stiefel_star = stiefel0
        beta_star = beta
        identity_lag = np.eye(self.lag, self.lag)
        identity_factors = np.eye(self.n_factors, self.n_factors)
        factors = range(self.n_factors)
        for date in range(250, 754):
            print(date)
            return_date = X_train.loc[:, str(date)].to_numpy()
            # print(f"return_date shape: {return_date.shape}")
            lagged_returns = X_train.loc[
                [(str(date), i) for i in range(50)], :
            ].to_numpy()
            # print(f"lagged_returns shape: {lagged_returns.shape}")
            prediction = lagged_returns @ stiefel @ beta.reshape(-1, 1)
            # print(f"prediction shape: {prediction.shape}")
            # print(f"prediction.T shape: {prediction.T.shape}")
            returns = (
                return_date.T @ lagged_returns / np.sqrt(np.linalg.norm(return_date, 2))
            )
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
            stiefel = self.project(stiefel_temp)
            # stiefel = stiefel_temp @ inv(sqrtm(stiefel_temp.T @ stiefel_temp))
            # print(f"beta shape: {beta.shape}")

            m = self.metric_train(stiefel, beta, X_train, Y_train)

            if m > max_metric:
                print(f"date:{date}", "metric_train:", m)
                max_metric = m
                stiefel_star = stiefel
                beta_star = beta

        return stiefel_star, beta_star