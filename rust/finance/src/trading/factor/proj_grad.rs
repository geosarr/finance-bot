#[cfg(test)]
mod test {
    use crate::loader::Loader;
    use crate::trading::factor::benchmark::BenchMark;
    use crate::trading::factor::preprocessor::{DataType, Preprocessor};
    use crate::trading::factor::utils::{calc_metric, norm_matrix, norm_vec, svd, MatrixType};
    use ndarray::parallel::prelude::*;
    use ndarray::{concatenate, s, Array, Axis};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::{rand::SeedableRng, RandomExt};
    use polars::prelude::ScanArgsParquet;
    use rand_chacha::ChaCha20Rng;
    use std::time::{Duration, Instant};
    #[test]
    fn test_code() {
        let start = Instant::now();
        let path = "../../data/returns.parquet";
        let loader = Loader::init(path);
        let data = loader.load_parquet(ScanArgsParquet::default());
        let max_lag = 250;
        let preprocessor = Preprocessor::init(max_lag);
        let (x_train, y_train) = preprocessor.split_x_y_ndarray(data, DataType::RETURN, 1);
        let duration = start.elapsed();
        println!("Time elapsed data processing is: {:?}", duration);
        // train(&x_train, &y_train);
        let eps = 1e-6;
        let max_lag = 250;
        let n_factors = 3;
        let n_iter = 150;
        let tol = 1e-6;
        let random_state = 12345;
        let stiefel_step = 0.1;
        let beta_step = 0.05;
        let start = Instant::now();
        let mut benchmark_model = BenchMark::init(100, tol, random_state, max_lag, n_factors);
        let (stiefel, beta) = benchmark_model.parallel_train(&x_train, &y_train, Some(5));
        // let duration = start.elapsed();
        // println!("Time elapsed parallel training: {:?}", duration);
        let start = Instant::now();
        // let mut stiefel = Array::ones((max_lag, n_factors));
        // let mut beta = Array::ones((n_factors, 1)); // .reshape((-1, 1));
        //                                             // println!("stiefel: \n{:?}", stiefel);
        //                                             // println!("beta: \n{:?}", beta);
        //                                             // let (stiefel, beta) = benchmark_model.sequential_train(&x_train, &y_train);
        //                                             // let duration = start.elapsed();
        //                                             // println!("Time elapsed sequential training: {:?}", duration);
        let mut stiefel_star = Array::zeros((max_lag, n_factors));
        let mut beta_star = Array::zeros((n_factors, 1));
        let mut max_metric = calc_metric(eps, &stiefel, &beta, &x_train, &y_train);
        let identity_lag: MatrixType<f64> = Array::eye(max_lag);
        let identity_factors: MatrixType<f64> = Array::eye(n_factors);
        let n_assets = y_train.shape()[0];
        let n_dates = x_train.shape()[0] / n_assets;
        let mut rng = ChaCha20Rng::seed_from_u64(random_state);
        let random_dates =
            Array::random_using(n_iter, Uniform::new_inclusive(0, n_dates - 1), &mut rng);
        let factors: Vec<usize> = (0..n_factors).collect();
        for (pos, date) in random_dates.iter().enumerate() {
            if pos > 10 {
                break;
            }
            // println!("xtrain: \n{:?}", x_train);
            // for date in &[493] {
            let return_date = y_train.slice(s![.., *date]);
            // println!("return_date: \n{:?}\n", return_date);
            let start = *date * n_assets;
            let end = (*date + 1) * n_assets;
            let lagged_returns = x_train.slice(s![start..end, ..]);
            // println!("lagged_returns: \n{:?}\n", lagged_returns);
            let prediction = lagged_returns.dot(&stiefel).dot(&beta);
            // println!("prediction: \n{:?}\n", prediction);
            // println!("{date}");
            // println!("{:?}", stiefel.shape());
            let temp_ret = return_date.t().dot(&lagged_returns);
            // println!("temp_ret: \n{:?}\n", temp_ret);
            let norm = norm_vec(return_date).sqrt();
            // println!("norm: \n{:?}\n", norm);
            let returns = temp_ret / norm;
            // println!("returns: \n{:?}\n", returns);

            // println!("{:?}", returns);
            // Gradients.
            let grad_stiefel = factors
                .iter()
                .map(|factor| {
                    let derivative = returns.dot(
                        &(beta[[*factor, 0]] * &identity_lag
                            - beta[[*factor, 0]]
                                * &stiefel.dot(&beta).dot(&prediction.t()).dot(&lagged_returns)
                                / norm_matrix(&prediction).sqrt().powi(3)),
                    );
                    // println!("ZERO:\n{:?}", derivative);
                    let len = derivative.shape()[0];
                    let derivative = derivative.into_shape((len, 1)).unwrap();
                    // println!("UN:\n{:?}", derivative);
                    derivative
                })
                .collect::<Vec<MatrixType<f64>>>();
            // println!("\ngrad 0: \n{:?}", &grad_stiefel[0]);
            let grad_stiefel: Vec<_> = grad_stiefel
                .iter()
                .map(|ret| ret.slice(s![.., ..]))
                .collect();
            let grad_stiefel =
                concatenate(Axis(1), &grad_stiefel).expect("Failed to get grad Stiefel");
            // println!("grad stiefel:\n{:?}", grad_stiefel);

            let grad_beta = returns.dot(&stiefel).dot(
                &(&identity_factors
                    - &beta.dot(&prediction.t()).dot(&lagged_returns).dot(&stiefel)
                        / norm_matrix(&prediction).sqrt().powi(3)),
            );
            let len = grad_beta.shape()[0];
            let grad_beta = grad_beta.into_shape((len, 1)).unwrap();
            // println!("grad beta:\n{:?}", grad_beta);
            // println!("\n{:?}", beta.shape());
            // println!("\n{:?}", grad_beta.shape());
            // Updates.
            let beta = &beta + beta_step * &grad_beta;
            let mut stiefel = &stiefel + stiefel_step * &grad_stiefel;
            let (u, s, v) = svd(&stiefel, None, 1000, 123);
            // println!("u: \n{:?}", u);
            // println!("s: \n{:?}", s);
            // println!("v: \n{:?}", v);
            stiefel = u.dot(&v);

            // println!("{:?}", beta.shape());
            // println!("{:?}", stiefel.shape());
            let metric = calc_metric(eps, &stiefel, &beta, &x_train, &y_train);
            println!("{metric}");
            if metric > max_metric {
                println!("Best metric: {metric}");
                stiefel_star = stiefel;
                beta_star = beta;
                max_metric = metric;
            }
        }
    }

    fn train(x_train: &MatrixType<f64>, y_train: &MatrixType<f64>) {
        let eps = 1e-6;
        let max_lag = 250;
        let n_factors = 10;
        let n_iter = 150;
        let tol = 1e-6;
        let random_state = 12345;
        let stiefel_step = 0.1;
        let beta_step = 0.05;
        let mut benchmark_model = BenchMark::init(40, tol, random_state, max_lag, n_factors);
        let (stiefel, beta) = benchmark_model.parallel_train(x_train, y_train, Some(5));
        // let (stiefel, beta) = benchmark_model.sequential_train(x_train, y_train);
        // let mut stiefel_star = Array::zeros((max_lag, n_factors));
        // let mut beta_star = Array::zeros((n_factors, 1));
        // let mut max_metric = calc_metric(eps, &stiefel, &beta, x_train, y_train);
        // let identity_lag: MatrixType<f64> = Array::eye(max_lag);
        // let identity_factors: MatrixType<f64> = Array::eye(n_factors);
        // let n_assets = y_train.shape()[0];
        // let n_dates = x_train.shape()[0] / n_assets;
        // let mut rng = ChaCha20Rng::seed_from_u64(random_state);
        // let random_dates =
        //     Array::random_using(n_iter, Uniform::new_inclusive(0, n_dates - 1), &mut rng);
        // let factors: Vec<usize> = (0..n_factors).collect();
        // for date in &random_dates {
        //     let return_date = y_train.slice(s![.., *date]);
        //     let start = *date * n_assets;
        //     let end = (*date + 1) * n_assets;
        //     let lagged_returns = x_train.slice(s![start..end, ..]);
        //     let prediction = lagged_returns.dot(&stiefel).dot(&beta);
        //     // println!("{date}");
        //     // println!("{:?}", stiefel.shape());
        //     let returns = return_date.t().dot(&lagged_returns) / norm_vec(return_date);

        //     // println!("{:?}", returns);
        //     // Gradients.
        //     let grad_stiefel = factors
        //         .iter()
        //         .map(|factor| {
        //             let derivative = returns.dot(
        //                 &(beta[[*factor, 0]] * &identity_lag
        //                     - beta[[*factor, 0]]
        //                         * &stiefel.dot(&beta).dot(&prediction.t()).dot(&lagged_returns)),
        //             ) / norm_matrix(&prediction).powi(3);
        //             // println!("ZERO:\n{:?}", derivative);
        //             let len = derivative.shape()[0];
        //             let derivative = derivative.into_shape((len, 1)).unwrap();
        //             // println!("UN:\n{:?}", derivative);
        //             derivative
        //         })
        //         .collect::<Vec<MatrixType<f64>>>();
        //     let grad_stiefel: Vec<_> = grad_stiefel
        //         .iter()
        //         .map(|ret| ret.slice(s![.., ..]))
        //         .collect();
        //     let grad_stiefel =
        //         concatenate(Axis(1), &grad_stiefel).expect("Failed to get grad Stiefel");

        //     let grad_beta = returns.dot(&stiefel).dot(
        //         &(&identity_factors
        //             - &beta.dot(&prediction.t()).dot(&lagged_returns).dot(&stiefel)),
        //     ) / norm_matrix(&prediction).powi(3);
        //     let len = grad_beta.shape()[0];
        //     let grad_beta = grad_beta.into_shape((len, 1)).unwrap();
        //     // println!("\n{:?}", beta.shape());
        //     // println!("\n{:?}", grad_beta.shape());
        //     // Updates.
        //     let beta = &beta + beta_step * &grad_beta;
        //     let stiefel = &stiefel + stiefel_step * &grad_stiefel;
        //     // println!("{:?}", beta.shape());

        //     // println!("{:?}", beta.shape());
        //     // println!("{:?}", stiefel.shape());
        //     let metric = calc_metric(eps, &stiefel, &beta, x_train, y_train);
        //     if metric > max_metric {
        //         println!("Best metric: {metric}");
        //         stiefel_star = stiefel;
        //         beta_star = beta;
        //         max_metric = metric;
        //     }
        // }
    }
}
