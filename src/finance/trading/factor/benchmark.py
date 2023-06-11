from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple
from tqdm.auto import tqdm
from finance.trading.factor.__model import Model


@dataclass
class Benchmark(Model):
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
        if self.max_metric > -1.0:
            # Reset the maximum metric at each run
            self.max_metric = -1.0
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
