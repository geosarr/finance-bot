from dataclasses import dataclass
import numpy as np


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
    # Apply Gram-Schmidt algorithm
    # to the columns of the gaussian matrix
    stiefel = np.linalg.qr(gaussian)[0]
    return stiefel


def is_orthonormal(eps: float, stiefel: np.ndarray) -> bool:
    n_factors = stiefel.shape[1]
    error = np.abs(stiefel.T @ stiefel - np.eye(n_factors))
    return not np.any(error > eps)


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.sqrt(np.sum(matrix**2, 0))


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
    n_assets = Y_train.shape[0]
    Y_pred = np.reshape(Y_pred, (n_assets, -1), "F")
    Y_pred = normalize(Y_pred)
    Y_true = normalize(Y_train)
    mean_overlap = np.mean(np.sum(Y_true * Y_pred, 0))
    return mean_overlap
