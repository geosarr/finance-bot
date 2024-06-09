#[cfg(test)]
mod unit_test;
use crate::trading::factor::utils::{calc_metric, get_orthonormal, MatrixType};
use ndarray::{concatenate, Array, Axis};
use ndarray_linalg::solve::Inverse;
use std::cmp::{Ordering, PartialOrd};
use std::sync::{Arc, Mutex};
use threadpool;

#[derive(Debug)]
struct ParamResult {
    stiefel: MatrixType<f64>,
    beta: MatrixType<f64>,
}
impl ParamResult {
    pub fn init(stiefel: MatrixType<f64>, beta: MatrixType<f64>) -> Self {
        Self { stiefel, beta }
    }
}

/// Stores the hyperparameters of the benchmark model
pub struct BenchMark {
    n_iter: usize,
    eps: f64,
    random_state: u64,
    n_lags: usize,
    n_factors: usize,
    max_metric: f64,
}

#[derive(Debug)]
struct BenchMarkResult {
    // Stiefel and beta parameters
    params: Option<ParamResult>,
    // Performance of the parameters
    metric: f64,
    // Seed used to generate the parameters
    random_state: u64,
    // Arbitrarily high number to convert f64
    // metric to usize (should depend on the use case)
    // max_decimal: f64,
    // Number of threads
}

impl BenchMarkResult {
    pub fn init(params: Option<ParamResult>, metric: f64, random_state: u64) -> Self {
        Self {
            params,
            metric,
            random_state,
            // max_decimal: 100000., // arbitrary, as long as it is high enough
        }
    }
    pub fn random_state(&self) -> u64 {
        self.random_state
    }
    pub fn metric(&self) -> f64 {
        self.metric
    }
    pub fn into_params_metric(self) -> (Option<MatrixType<f64>>, Option<MatrixType<f64>>, f64) {
        if let Some(params) = self.params {
            (Some(params.stiefel), Some(params.beta), self.metric)
        } else {
            (None, None, self.metric)
        }
    }
}
impl BenchMark {
    pub fn init(
        n_iter: usize,
        eps: f64,
        random_state: u64,
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
    pub fn max_metric(&self) -> f64 {
        self.max_metric
    }
    fn reset_max_metric(&mut self) {
        if self.max_metric > -1.0 {
            // Reset to the minimum
            // value of the cosine metric, i.e -1
            self.max_metric = -1.0;
        }
    }
    pub fn parallel_train(
        &mut self,
        x_train: &MatrixType<f64>,
        y_train: &MatrixType<f64>,
        num_threads: Option<usize>,
    ) -> (MatrixType<f64>, MatrixType<f64>) {
        self.reset_max_metric();
        return parallel_train(
            self.n_iter,
            self.eps,
            self.n_lags,
            self.n_factors,
            self.random_state,
            x_train,
            y_train,
            &mut self.max_metric,
            num_threads,
        );
    }
    pub fn sequential_train(
        &mut self,
        x_train: &MatrixType<f64>,
        y_train: &MatrixType<f64>,
    ) -> (MatrixType<f64>, MatrixType<f64>) {
        self.reset_max_metric();
        return sequential_train(
            self.n_iter,
            self.eps,
            self.n_lags,
            self.n_factors,
            self.random_state,
            x_train,
            y_train,
            &mut self.max_metric,
        );
    }
}

/// Trains a random Stiefel model in parallel
pub fn parallel_train(
    n_iter: usize,
    eps: f64,
    n_lags: usize,
    n_factors: usize,
    random_state: u64,
    x_train: &MatrixType<f64>,
    y_train: &MatrixType<f64>,
    max_metric: &mut f64,
    num_threads: Option<usize>,
) -> (MatrixType<f64>, MatrixType<f64>) {
    // Default number of threads
    let mut _num_threads = 10;
    if let Some(num) = num_threads {
        _num_threads = num;
    }
    let pool = threadpool::Builder::new().num_threads(_num_threads).build();
    let handles = Arc::new(Mutex::new(Vec::new()));
    let targets = Arc::new(vectorize(y_train, 0));
    let xtrain = Arc::new(x_train.clone()); // may be memory intensive for some use cases
    let ytrain = Arc::new(y_train.clone()); // may take memory intensive for some use cases
    let maxmetric = Arc::new(Mutex::new(max_metric.clone()));
    for iter in 0..n_iter {
        let handles = Arc::clone(&handles);
        let targets = Arc::clone(&targets);
        let xtrain = Arc::clone(&xtrain);
        let ytrain = Arc::clone(&ytrain);
        let maxmetric = Arc::clone(&maxmetric);
        let seed = random_state + (iter as u64);
        pool.execute(move || {
            let result = candidate_params(
                eps, n_lags, n_factors, seed, false, &xtrain, &ytrain, &targets,
            );
            let mut maxmetric = maxmetric.lock().unwrap();
            println!("Iteration {iter}");
            if *maxmetric < result.metric() {
                println!("Best parameters found with seed {seed}.");
                let mut handles = handles.lock().unwrap();
                *maxmetric = result.metric();
                handles.push(result);
            }
        });
    }
    pool.join();
    // Contains only the random states and the metric of the candidate parameters
    let results = handles.lock().unwrap();
    let best_seed = &results.last().unwrap().random_state();
    let (stiefel_star, beta_star, metric_star) = candidate_params(
        eps, n_lags, n_factors, *best_seed, true, x_train, y_train, &targets,
    )
    .into_params_metric();
    *max_metric = metric_star;
    return (stiefel_star.unwrap(), beta_star.unwrap());
}

/// Trains a random Stiefel model sequentially
pub fn sequential_train(
    n_iter: usize,
    eps: f64,
    n_lags: usize,
    n_factors: usize,
    random_state: u64,
    x_train: &MatrixType<f64>,
    y_train: &MatrixType<f64>,
    max_metric: &mut f64,
) -> (MatrixType<f64>, MatrixType<f64>) {
    let mut stiefel_star = Array::zeros((n_lags, n_factors));
    let mut beta_star = Array::zeros((n_factors, 1));
    let targets = vectorize(y_train, 0);
    // Trajectory of best parameters
    let _: Vec<_> = (0..n_iter)
        .filter_map(|iter| {
            println!("Iteration {iter}");
            test_params(
                eps,
                n_lags,
                n_factors,
                random_state + (iter as u64),
                x_train,
                y_train,
                &targets,
                max_metric,
                &mut stiefel_star,
                &mut beta_star,
            )
        })
        .collect();
    return (stiefel_star, beta_star);
}

/// Tests a candidate pair of random
/// Stiefel matrix and a linear regression parameter.
fn test_params(
    eps: f64,
    n_lags: usize,
    n_factors: usize,
    seed: u64,
    x_train: &MatrixType<f64>,
    y_train: &MatrixType<f64>,
    targets: &MatrixType<f64>,
    max_metric: &mut f64,
    stiefel_star: &mut MatrixType<f64>,
    beta_star: &mut MatrixType<f64>,
) -> Option<u64> {
    let (stiefel, beta, metric) = candidate_params(
        eps, n_lags, n_factors, seed, true, x_train, y_train, targets,
    )
    .into_params_metric();
    let mut flag_better_params = None;
    if metric > *max_metric {
        *max_metric = metric;
        flag_better_params = Some(seed);
        println!("Best parameters found with seed {seed}.");
        *stiefel_star = stiefel.unwrap();
        *beta_star = beta.unwrap();
    }
    return flag_better_params;
}

/// Gets a candidate Stiefel matrix and a linear
/// regression coefficient along with the associated metrics.
fn candidate_params(
    eps: f64,
    n_lags: usize,
    n_factors: usize,
    random_state: u64,
    with_params: bool,
    x_train: &MatrixType<f64>,
    y_train: &MatrixType<f64>,
    targets: &MatrixType<f64>,
) -> BenchMarkResult {
    let stiefel = get_orthonormal(n_lags, n_factors, random_state);
    let beta = fit_beta(&stiefel, x_train, targets);
    let metric = calc_metric(eps, &stiefel, &beta, x_train, y_train);
    let result = if with_params {
        BenchMarkResult::init(Some(ParamResult::init(stiefel, beta)), metric, random_state)
    } else {
        BenchMarkResult::init(None, metric, random_state)
    };
    return result;
}

/// Fits a linear regression coefficient.
fn fit_beta(
    stiefel: &MatrixType<f64>,
    x_train: &MatrixType<f64>,
    y_train: &MatrixType<f64>,
) -> MatrixType<f64> {
    let predictor = x_train.dot(stiefel);
    return Inverse::inv(&predictor.t().dot(&predictor))
        .unwrap()
        .dot(&predictor.t())
        .dot(y_train);
}

/// Transforms `data` to a vector.
fn vectorize(data: &MatrixType<f64>, axis: usize) -> MatrixType<f64> {
    let shape = data.shape();
    let columns_of_data: Vec<_> = (0..shape[1]).map(|num_col| data.column(num_col)).collect();
    return concatenate(Axis(axis), &columns_of_data)
        .expect("Failed to vectorize.")
        .into_shape((shape[0] * shape[1], 1))
        .expect("Failed to put into shape.");
}

// impl PartialOrd for BenchMarkResult {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }
// impl Ord for BenchMarkResult {
//     fn cmp(&self, other: &Self) -> Ordering {
//         ((self.max_decimal * self.metric) as usize)
//             .cmp(&((other.max_decimal * other.metric) as usize))
//     }
// }
// impl PartialEq for BenchMarkResult {
//     fn eq(&self, other: &Self) -> bool {
//         (self.max_decimal * self.metric) as usize == (self.max_decimal * other.metric) as usize
//     }
// }
// impl Eq for BenchMarkResult {}
