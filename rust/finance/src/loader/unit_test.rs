#[cfg(test)]

mod test {
    use super::super::*;
    use polars::prelude::*;

    #[test]
    fn test_load_csv() {
        let path = "../../data/returns.csv";
        let loader = Loader::init(path);
        let data = loader.load_csv(b';', true).collect().unwrap();
        assert_eq!(data.width(), 50);
        assert_eq!(data.height(), 754);
        println!("{:?}", data);
    }

    #[test]
    fn test_load_parquet() {
        let path = "../../data/returns.parquet";
        let loader = Loader::init(path);
        let data = loader
            .load_parquet(ScanArgsParquet::default())
            .collect()
            .unwrap();
        assert_eq!(data.width(), 50);
        assert_eq!(data.height(), 754);
        println!("{:?}", data);
    }
}
