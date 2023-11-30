#[cfg(test)]
mod unit_test;
use crate::trading::factor::utils::{calc_metric, get_orthonormal, MatrixType, VecType};
use ndarray::{concatenate, Array, Axis};
use ndarray_linalg::solve::Inverse;

pub struct BenchMark {
    n_iter: usize,
    eps: f64,
    random_state: usize,
    n_lags: usize,
    n_factors: usize,
    max_metric: f64,
}

impl BenchMark {
    pub fn init(
        n_iter: usize,
        eps: f64,
        random_state: usize,
        n_lags: usize,
        n_factors: usize,
    ) -> Self {
        Self {
            n_iter,
            eps,
            random_state,
            n_lags,
            n_factors,
            max_metric: -1.0,
        }
    }
    /// Fits a linear regression coefficient.
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
    /// Trains a random Stiefel model
    pub fn train(
        &mut self,
        x_train: &MatrixType,
        y_train: &MatrixType,
    ) -> (MatrixType, MatrixType) {
        if self.max_metric > -1.0 {
            // Reset the maximum metric at each run
            self.max_metric = -1.0;
        }
        let mut stiefel_star = Array::zeros((self.n_lags, self.n_factors));
        let mut beta_star = Array::zeros((self.n_factors, 1));
        let targets = self.vectorize(y_train, 0);
        for iter in 0..self.n_iter {
            // TODO add random state in get_othonormal.
            let stiefel = get_orthonormal(self.n_lags, self.n_factors);
            let beta = self.fit_beta(&stiefel, x_train, &targets);
            let metric = calc_metric(self.eps, &stiefel, &beta, x_train, y_train);
            if metric > self.max_metric {
                self.max_metric = metric;
                println!("Best parameters found at iteration {iter}.");
                stiefel_star = stiefel;
                beta_star = beta;
            }
        }
        return (stiefel_star, beta_star);
    }
    /// Transforms `data` to a vector.
    fn vectorize(&self, data: &MatrixType, axis: usize) -> MatrixType {
        let shape = data.shape();
        let columns_of_data: Vec<_> = (0..shape[1]).map(|num_col| data.column(num_col)).collect();
        return concatenate(Axis(axis), &columns_of_data)
            .expect("Failed to vectorize.")
            .into_shape((shape[0] * shape[1], 1))
            .expect("Failed to put into shape.");
    }
}
