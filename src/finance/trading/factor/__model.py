from dataclasses import dataclass
import numpy as np
import pandas as pd
from numba import jit, float64, boolean, int64


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
        return get_orthonormal(self.lag, self.n_factors)

    def metric_train(
        self,
        stiefel: np.ndarray,
        beta: np.ndarray,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ):
        return calc_metric(self.eps, stiefel, beta, X_train, Y_train)


def get_orthonormal(n_lags: int, n_factors: int) -> np.ndarray:
    gaussian = np.random.randn(n_lags, n_factors)
    # Apply Gram-Schmidt algorithm to the columns of the matrix gaussian
    stiefel = np.linalg.qr(gaussian)[0]
    return stiefel


# @jit([boolean(float64, float64[:, :])], nopython=True)
def is_orthonormal(eps: float, stiefel: np.ndarray) -> bool:
    n_factors = stiefel.shape[1]
    error = np.abs(stiefel.T @ stiefel - np.eye(n_factors))
    return not np.any(error > eps)


# @jit([float64[:, :](float64[:, :])], nopython=True)
def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.sqrt(
        np.sum(matrix**2, 0)
    )  # np.linalg.norm(matrix, ord=norm, axis=0)


# @jit(
#     [float64(float64, float64[:, :], float64[:], float64[:, :], float64[:, :])],
#     nopython=True,
# )
def calc_metric(
    eps: float,
    stiefel: np.ndarray,
    beta: np.ndarray,
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> float:
    if not is_orthonormal(eps, stiefel):
        return -1.0

    Y_pred = X_train @ stiefel @ beta
    # Y_pred = Y_pred.unstack().T
    # print(Y_pred.shape)
    n_assets = Y_train.shape[0]
    # print("assets", n_assets)
    Y_pred = np.reshape(Y_pred, (n_assets, -1), "F")
    # print("Y_pred\n", Y_pred)

    Y_true = normalize(Y_train)  # Y_train.div(np.sqrt((Y_train**2).sum()), 1)
    # print("Y_true normalized\n", Y_true)

    Y_pred = normalize(Y_pred)  # Y_pred.div(np.sqrt((Y_pred**2).sum()), 1)
    # print("Y_pred normalized\n", Y_pred)

    mean_overlap = np.mean(np.sum(Y_true * Y_pred, 0))
    # print("mean_overlap\n", mean_overlap)

    return mean_overlap
