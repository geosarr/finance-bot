use polars::series::Series;

pub fn pct_change(prices: &Series) -> Series {
    (prices / &prices.shift(1)) - 1
}
