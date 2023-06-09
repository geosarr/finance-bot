from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple


@dataclass
class Model:
    lag: int = 250
    n_factors: int = 10
    eps: float = 1e-6

    # Taken and adapted from QRTChallenge
    def random_stiefel(self) -> np.ndarray:
        gaussian = np.random.randn(self.lag, self.n_factors)
        # Apply Gram-Schmidt algorithm to the columns of the matrix gaussian
        stiefel = np.linalg.qr(gaussian)[0]
        return stiefel

    def check_orthonormality(self, stiefel: np.ndarray) -> bool:
        n_factors = stiefel.shape[1]
        error = pd.DataFrame(stiefel.T @ stiefel - np.eye(n_factors)).abs()
        return any(error.unstack() > self.eps)

    def fit_beta(
        stiefel: np.ndarray, X_train: pd.DataFrame, Y_train: pd.DataFrame
    ) -> np.ndarray:
        # the dataframe of the factors created with PreProcess
        predictors: pd.DataFrame = X_train @ stiefel
        targets: pd.DataFrame = Y_train.T.stack()
        beta: pd.DataFrame = (
            np.linalg.inv(predictors.T @ predictors) @ predictors.T @ targets
        )

        return beta.to_numpy()

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

    def train(
        self, X_train: pd.DataFrame, Y_train: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.random_state)
        max_metric = -1
        for iteration in range(self.n_iter):
            # Generate a uniform random Stiefel matrix and fit beta
            # with minimal mean square prediction error on the training data set
            stiefel = self.random_stiefel()
            beta = self.fit_beta(stiefel)
            # Compute the metric on the training set and keep the best result
            m = self.metric_train(stiefel, beta, X_train, Y_train)
            if m > max_metric:
                print(iteration, "metric_train:", m)
                max_metric = m
                stiefel_star = stiefel
                beta_star = beta
        return stiefel_star, beta_star
