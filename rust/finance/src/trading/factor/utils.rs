#[cfg(test)]
mod unit_test;
use ndarray::{Array, ArrayBase, Axis, Dim, OwnedRepr};
use polars::series::Series;

pub fn pct_change(prices: &Series) -> Series {
    (prices / &prices.shift(1)) - 1
}

pub type MatrixType = ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>;
pub fn is_orthonormal(eps: f64, stiefel: &MatrixType) -> bool {
    let eye: MatrixType = Array::eye(stiefel.shape()[1]);
    let error = (stiefel.t().dot(stiefel) - eye).map(|x| x.abs());
    return !error.iter().any(|x| x > &eps);
}

pub fn normalize(matrix: &MatrixType) -> MatrixType {
    let norm = matrix
        .map(|x| x.powi(2))
        // .collect::<MatrixType>()
        .sum_axis(Axis(0))
        .map(|x| x.sqrt());
    return matrix / norm;
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
        let y_pred = x_train.dot(stiefel).dot(beta);
        let shape = y_pred.shape();
        let (nb_assets, nb_dates_to_predict) = (shape[0], shape[1]);
        let y_pred = y_pred.into_shape((nb_assets, nb_dates_to_predict)).unwrap();
        let y_pred = normalize(&y_pred);
        let y_true = normalize(y_train);
        let mean_overlap = (y_pred * y_true).sum_axis(Axis(0)).mean();
        return mean_overlap.unwrap();
    }
}
