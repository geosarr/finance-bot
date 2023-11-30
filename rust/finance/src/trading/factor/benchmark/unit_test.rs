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
        let mut benchmark_model = BenchMark::init(20, 1e-6, 100, 250, 10);
        let (stiefel_star, beta_star) = benchmark_model.train(&x_train, &y_train);
        println!("{:?}", stiefel_star);
        println!("{:?}", beta_star);
    }
}
