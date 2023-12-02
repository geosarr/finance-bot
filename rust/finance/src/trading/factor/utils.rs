#[cfg(test)]
mod unit_test;
use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, Axis, Dim};
use ndarray_rand::{rand::SeedableRng, rand_distr::StandardNormal, RandomExt};
use polars::series::Series;
use rand_chacha::ChaCha20Rng;

pub type MatrixType = ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>;
pub type VecType = ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>;
type VecViewType<'a> = ArrayBase<ndarray::ViewRepr<&'a f64>, Dim<[usize; 1]>>;

pub fn pct_change(prices: &Series) -> Series {
    (prices / &prices.shift(1)) - 1
}

pub fn is_orthonormal(eps: f64, stiefel: &MatrixType) -> bool {
    let eye: MatrixType = Array::eye(stiefel.shape()[1]);
    let error = (stiefel.t().dot(stiefel) - eye).map(|x| x.abs());
    return !error.iter().any(|x| x > &eps);
}

pub fn schwartz_rutishauser_qr(matrix: &MatrixType) -> (MatrixType, MatrixType) {
    let mut q = matrix.clone();
    let shape = matrix.shape();
    let (_, n) = (shape[0], shape[1]);
    let mut r = Array::<f64, _>::zeros((n, n));
    for k in 0..n {
        for i in 0..k {
            r[[i, k]] = q.column(i).t().dot(&q.column(k));
            (q.column(k).to_owned() - r[[i, k]] * q.column(i).to_owned())
                .assign_to(q.slice_mut(s![.., k]));
        }
        r[[k, k]] = norm_vec(q.slice(s![.., k]));
        (q.column(k).to_owned() / r[[k, k]]).assign_to(q.slice_mut(s![.., k]));
    }
    return (-q, -r);
}

pub fn get_orthonormal(n_lags: usize, n_factors: usize, seed: u64) -> MatrixType {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let gaussian: MatrixType = Array::random_using((n_lags, n_factors), StandardNormal, &mut rng);
    return schwartz_rutishauser_qr(&gaussian).0;
}

pub fn normalize(matrix: &MatrixType, axis: usize) -> MatrixType {
    let norm = matrix
        .map(|x| x.powi(2))
        .sum_axis(Axis(axis))
        .map(|x| x.sqrt());
    return matrix / norm;
}

pub fn norm_vec<'a>(vec: VecViewType<'a>) -> f64 {
    return vec.map(|x| x.powi(2)).sum().sqrt();
}

pub fn calc_metric(
    eps: f64,
    stiefel: &MatrixType,
    beta: &MatrixType,
    x_train: &MatrixType,
    y_train: &MatrixType,
) -> f64 {
    if !is_orthonormal(eps, &stiefel) {
        return -1.0;
    } else {
        let shape = y_train.shape();
        let (nb_assets, nb_dates_to_predict) = (shape[0], shape[1]);
        let y_pred = x_train.dot(stiefel).dot(beta);
        let y_pred = y_pred.into_shape((nb_dates_to_predict, nb_assets)).unwrap();
        let y_pred = y_pred.t().to_owned();
        let y_pred = normalize(&y_pred, 0);
        let y_true = normalize(y_train, 0);
        let mean_overlap = (y_pred * y_true).sum_axis(Axis(0)).mean();
        return mean_overlap.unwrap();
    }
}
