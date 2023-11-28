#[cfg(test)]
mod test {
    use super::super::*;
    use crate::loader::Loader;

    #[test]
    fn test_preprocessor() {
        let path = "../../data/returns.parquet";
        let loader = Loader::init(path);
        let data = loader.load_parquet(ScanArgsParquet::default());
        let max_lag = 250;
        let preprocessor = Preprocessor::init(max_lag);
        let (x_train, y_train) = preprocessor.split_x_y_ndarray(data, DataType::RETURN, 1);
        assert_eq!(x_train.nrows(), 50 * (754 - max_lag));
        assert_eq!(x_train.ncols(), max_lag);
        assert_eq!(y_train.nrows(), 50);
        assert_eq!(y_train.ncols(), 754 - max_lag);
    }
}
