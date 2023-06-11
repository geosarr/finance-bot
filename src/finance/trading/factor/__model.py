from dataclasses import dataclass
import numpy as np
import pandas as pd


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
