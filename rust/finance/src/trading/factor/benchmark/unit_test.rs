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
        let random_state = 12345;
        let mut benchmark_model = BenchMark::init(n_iter, tol, random_state, max_lag, n_factors);
        let (par_stiefel_star, par_beta_star) =
            benchmark_model.parallel_train(&x_train, &y_train, Some(5));
        // let (seq_stiefel_star, seq_beta_star) = benchmark_model.train(&x_train, &y_train, false);
        // assert_eq!(par_stiefel_star.nrows(), max_lag);
        // assert_eq!(par_stiefel_star.ncols(), n_factors);
        // assert_eq!(par_stiefel_star, seq_stiefel_star);
        // assert_eq!(par_beta_star, seq_beta_star);
        // println!("{:?}", par_stiefel_star);
        // println!("{:?}", par_beta_star);
        // println!("{}", benchmark_model.max_metric());
    }
}
