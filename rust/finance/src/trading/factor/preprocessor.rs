#[cfg(test)]
mod unit_test;
use crate::trading::factor::utils::MatrixType;
use ndarray::concatenate;
use ndarray::prelude::*;
use polars::prelude::*;
type TrainingData = (MatrixType<f64>, MatrixType<f64>);
pub enum DataType {
    PRICE,
    RETURN,
}

pub struct Preprocessor {
    max_lag: usize,
}

impl Preprocessor {
    pub fn init(max_lag: usize) -> Self {
        Self { max_lag }
    }
    pub fn max_lag(&self) -> usize {
        self.max_lag
    }
    pub fn split_x_y_ndarray(
        &self,
        data: LazyFrame,
        data_type: DataType,
        period: i32,
    ) -> TrainingData {
        // Assets should be column-wise like
        // LazyFrame({"asset_1": [price_1, price_2, ...], "asset_2": [price_1, price_2, ...], ...})
        // or LazyFrame({"asset_1": [return_1, return_2, ...], "asset_2": [return_1, return_2, ...], ...})
        let returns = match data_type {
            DataType::PRICE => data.select([col("*") / col("*").shift(lit(period)) - lit(period)]),
            DataType::RETURN => data,
        };
        let returns = returns
            .collect()
            .expect("Failed to get returns")
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let nb_time_steps = returns.shape()[0];
        let mut x_train: Vec<ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>>> = (0
            ..nb_time_steps - self.max_lag)
            .map(|index| returns.slice(s![index..index + self.max_lag, ..]))
            .collect();
        let x_train = concatenate(Axis(1), &x_train[..])
            .expect("Failed to get x_train")
            .t()
            .to_owned();
        let y_train = returns.slice(s![self.max_lag.., ..]).t().to_owned();
        return (x_train.slice(s![.., ..;-1]).to_owned(), y_train);
    }
}
