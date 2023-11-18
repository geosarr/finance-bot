#[cfg(test)]
mod unit_test;
use polars::prelude::*;

pub struct Loader<'a> {
    path: &'a str,
}

impl<'a> Loader<'a> {
    pub fn init(path: &'a str) -> Self {
        Self { path }
    }
    pub fn path(&self) -> &str {
        self.path
    }
    pub fn load_csv(&self, sep: u8, header: bool) -> LazyFrame {
        if self.path.ends_with(".csv") {
            return LazyCsvReader::new(self.path)
                .with_separator(sep)
                .has_header(header)
                .finish()
                .expect("Failed to load file");
        } else {
            panic!("Only supports csv files.")
        }
    }
    pub fn load_parquet(&self, args: ScanArgsParquet) -> LazyFrame {
        if self.path.ends_with(".parquet") {
            return LazyFrame::scan_parquet(self.path, args).unwrap();
        } else {
            panic!("Only supports parquet files.")
        }
    }
}
