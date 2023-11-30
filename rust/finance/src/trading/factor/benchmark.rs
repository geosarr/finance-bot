use crate::trading::factor::utils::{get_orthonormal, MatrixType};
use ndarray_linalg::solve::Inverse;

pub struct BenchMark {
    n_iter: usize,
    random_state: usize,
    n_lags: usize,
    n_factors: usize,
    max_metric: f64,
}

impl BenchMark {
    pub fn init(n_iter: usize, random_state: usize, n_lags: usize, n_factors: usize) -> Self {
        Self {
            n_iter,
            random_state,
            n_lags,
            n_factors,
            max_metric: -1.0,
        }
    }
    pub fn fit_beta(
        &self,
        stiefel: &MatrixType,
        x_train: &MatrixType,
        y_train: &MatrixType,
    ) -> MatrixType {
        let predictor = x_train.dot(stiefel);
        return Inverse::inv(&predictor.t().dot(&predictor))
            .unwrap()
            .dot(&predictor.t())
            .dot(y_train);
    }
    pub fn train(&mut self, x_train: &MatrixType, y_train: &MatrixType) {
        if self.max_metric > -1.0 {
            // Reset the maximum metric at each run
            self.max_metric = -1.0
        } else {
            for iter in 0..self.n_iter {
                let stiefel = get_orthonormal(self.n_lags, self.n_factors);
                let beta = self.fit_beta(&stiefel, x_train, y_train);
            }
        }
    }
}
