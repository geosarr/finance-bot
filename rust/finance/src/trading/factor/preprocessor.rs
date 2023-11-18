// use polars::lazy::{dsl::col, frame::LazyFrame};

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
    // pub fn split_x_y(&self, data: LazyFrame, data_type: DataType) {
    // Assets should be column-wise like
    // DataFrame({"asset_1": [price_1, price_2, ...], "asset_2": [price_1, price_2, ...], ...})
    // or DataFrame({"asset_1": [return_1, return_2, ...], "asset_2": [return_1, return_2, ...], ...})
    // match data_type {
    //     DataType::PRICE => let df = data.select(col("*").pct_change()),
    //     DataType::RETURN => let df = data.collect().unwrap().T,
    // }
    // if data_type == DataType.PRICE:
    //     df = data.pct_change().dropna().reset_index(drop=True).T
    // elif data_type == DataType.RETURN:
    //     df = data.T
    // Y_train = df.loc[:, self.max_lag :].to_numpy()
    // iter_data = map(
    //     lambda pos: df.T.shift(pos + 1).stack(dropna=False), range(self.max_lag)
    // )
    // X_train = pd.concat(iter_data, axis=1).dropna().to_numpy()
    // return X_train, Y_train
    // }
}
