#[cfg(test)]
mod unit_test;
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use polars::series::Series;

pub fn pct_change(prices: &Series) -> Series {
    (prices / &prices.shift(1)) - 1
}

pub type MatrixType = ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>;
fn is_orthonormal(eps: f64, stiefel: MatrixType) -> bool {
    let eye: MatrixType = Array::eye(stiefel.shape()[1]);
    let error = (stiefel.t().dot(&stiefel) - eye).map(|x| x.abs());
    return !error.iter().any(|x| x > &eps);
}
