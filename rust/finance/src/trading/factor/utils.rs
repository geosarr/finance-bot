#[cfg(test)]
mod unit_test;
use ndarray::prelude::*;
use ndarray::{parallel::prelude::*, Array, ArrayBase, Axis, Dim};
use ndarray_linalg::Inverse;
use ndarray_rand::{rand::SeedableRng, rand_distr::StandardNormal, RandomExt};
use polars::series::Series;
use rand_chacha::ChaCha20Rng;

pub type MatrixType<T> = ArrayBase<ndarray::OwnedRepr<T>, Dim<[usize; 2]>>;
pub type MatrixViewType<'a, T> = ArrayBase<ndarray::ViewRepr<&'a T>, Dim<[usize; 2]>>;
pub type VecType<T> = ArrayBase<ndarray::OwnedRepr<T>, Dim<[usize; 1]>>;
type VecViewType<'a, T> = ArrayBase<ndarray::ViewRepr<&'a T>, Dim<[usize; 1]>>;

pub fn pct_change(prices: &Series) -> Series {
    (prices / &prices.shift(1)) - 1
}

pub fn is_orthonormal(eps: f64, stiefel: &MatrixType<f64>) -> bool {
    let eye: MatrixType<f64> = Array::eye(stiefel.shape()[1]);
    let error = stiefel.t().dot(stiefel) - eye;
    return error.par_iter().all(|x| x.abs() <= eps);
}

pub fn schwartz_rutishauser_qr(matrix: &MatrixType<f64>) -> (MatrixType<f64>, MatrixType<f64>) {
    let mut q = matrix.clone();
    let n = matrix.shape()[1];
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
    return (q, r);
}

pub fn svd(
    matrix: &MatrixType<f64>,
    nb_sv: Option<usize>,
    nb_iter: usize,
    seed: u64,
) -> (MatrixType<f64>, MatrixType<f64>, MatrixType<f64>) {
    let (n_orig, m_orig) = (matrix.shape()[0], matrix.shape()[1]);
    let mut k = std::cmp::min(n_orig, m_orig);
    if let Some(nb) = nb_sv {
        k = std::cmp::min(nb, k);
    }
    // let mat = matrix.clone();
    let mat = if n_orig > m_orig {
        matrix.t().dot(matrix)
    } else if n_orig < m_orig {
        matrix.dot(&matrix.t())
    } else {
        matrix.clone()
    };
    let (mut n, mut m) = (mat.shape()[0], mat.shape()[1]);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut q: MatrixType<f64> = Array::random_using((n, k), StandardNormal, &mut rng);
    let (mut q, mut r) = schwartz_rutishauser_qr(&q);
    for _ in 0..nb_iter {
        let z = mat.dot(&q);
        let q_prev = q;
        (q, r) = schwartz_rutishauser_qr(&z);
        let err = norm_matrix(&(&q - &q_prev));
        if err.powi(2) < 0.00001 {
            break;
        }
    }
    // println!("r: \n{:?}", r);
    let diag_r = r.diag().map(|x| x.sqrt());
    // println!("vec_diag_r: \n{:?}", diag_r);
    let mut singular_values: MatrixType<f64> = ArrayBase::from_diag(&diag_r);
    // println!("sing_val: \n{:?}", singular_values);
    let mut left_vecs = q.reversed_axes();
    // println!("left: \n{:?}", left_vecs);
    let mut right_vecs = left_vecs.clone();
    // println!("right: \n{:?}", right_vecs);
    if n_orig < m_orig {
        right_vecs = Inverse::inv(&singular_values)
            .unwrap()
            .dot(&left_vecs.t())
            .dot(matrix);
    } else if n_orig > m_orig {
        left_vecs = matrix
            .dot(&right_vecs.t())
            .dot(&Inverse::inv(&singular_values).unwrap());
    } else {
        singular_values = singular_values.map(|x| x.powi(2));
    }
    return (left_vecs, singular_values, right_vecs);
}
pub fn get_orthonormal(n_lags: usize, n_factors: usize, seed: u64) -> MatrixType<f64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let gaussian: MatrixType<f64> =
        Array::random_using((n_lags, n_factors), StandardNormal, &mut rng);
    return schwartz_rutishauser_qr(&gaussian).0;
}

pub fn normalize(matrix: &MatrixType<f64>, axis: usize) -> MatrixType<f64> {
    let norm = matrix
        .map(|x| x.powi(2))
        .sum_axis(Axis(axis))
        .map(|x| x.sqrt());
    return matrix / norm;
}

pub fn norm_vec<'a>(vec: VecViewType<'a, f64>) -> f64 {
    return vec.map(|x| x.powi(2)).sum().sqrt();
}

// pub fn norm_matrix<'a>(vec: MatrixViewType<'a>) -> f64 {
//     return vec.map(|x| x.powi(2)).sum().sqrt();
// }

pub fn norm_matrix(vec: &MatrixType<f64>) -> f64 {
    return vec.map(|x| x.powi(2)).sum().sqrt();
}

pub fn calc_metric(
    eps: f64,
    stiefel: &MatrixType<f64>,
    beta: &MatrixType<f64>,
    x_train: &MatrixType<f64>,
    y_train: &MatrixType<f64>,
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
