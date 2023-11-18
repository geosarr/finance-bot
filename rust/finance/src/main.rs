use finance::loader::Loader;
use polars::{lazy::frame::ScanArgsParquet, series::Series};

mod loader;

fn main() {
    let loader = Loader::init("../../data/prices.parquet");
    let data = loader.load_parquet(ScanArgsParquet::default());
    // println!("{:?}", data.collect().unwrap());
    println!(
        "{:?}",
        data.collect()
            .expect("")
            .apply("PERNOD RICARD SA", pct_change) // data.select([
                                                   //     col("PERNOD RICARD SA"),
                                                   //     (col("PERNOD RICARD SA") / col("PERNOD RICARD SA").shift(lit(1)) - lit(1))
                                                   //         .alias("return")
                                                   // ])
                                                   // .collect()
    );
}

pub fn pct_change(prices: &Series) -> Series {
    (prices / &prices.shift(1)) - 1
}
