#[cfg(test)]
mod test {
    use super::super::*;
    use crate::loader::Loader;
    use crate::trading::factor::preprocessor::{DataType, Preprocessor};
    use polars::prelude::ScanArgsParquet;

    #[test]
    fn test_benchmark() {
        let path = "../../data/returns.parquet";
        let loader = Loader::init(path);
        let data = loader.load_parquet(ScanArgsParquet::default());
        let max_lag = 250;
        let preprocessor = Preprocessor::init(max_lag);
        let (x_train, y_train) = preprocessor.split_x_y_ndarray(data, DataType::RETURN, 1);
        let n_factors = 10;
        let n_iter = 100;
        let tol = 1e-6;
        let random_sate = 12345;
        let (stiefel_star, beta_star) = parallel_train(
            n_iter,
            tol,
            max_lag,
            n_factors,
            random_sate,
            &x_train,
            &y_train,
        );
        let mut benchmark_model = BenchMark::init(n_iter, tol, random_sate, max_lag, n_factors);
        let (stiefel_star, beta_star) = benchmark_model.sequential_train(&x_train, &y_train);
        assert_eq!(stiefel_star.nrows(), max_lag);
        assert_eq!(stiefel_star.ncols(), n_factors);
        println!("{:?}", stiefel_star);
        println!("{:?}", beta_star);
        println!("{}", benchmark_model.max_metric());
    }
}
