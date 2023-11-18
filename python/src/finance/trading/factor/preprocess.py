from dataclasses import dataclass
import pandas as pd
from enum import Enum, auto


class DataType(Enum):
    PRICE = auto()
    RETURN = auto()


@dataclass
class PreProcess:
    max_lag: int = 50

    def split_x_y(self, data: pd.DataFrame, data_type: DataType = DataType.PRICE):
        # Assets should be column-wise like
        # pd.DataFrame({"asset_1": [price_1, price_2, ...], "asset_2": [price_1, price_2, ...], ...})
        # or pd.DataFrame({"asset_1": [return_1, return_2, ...], "asset_2": [return_1, return_2, ...], ...})
        if data_type == DataType.PRICE:
            df = data.pct_change().dropna().reset_index(drop=True).T
        elif data_type == DataType.RETURN:
            df = data.copy().T
        Y_train = df.loc[:, self.max_lag :].to_numpy()
        iter_data = map(
            lambda pos: df.T.shift(pos + 1).stack(dropna=False), range(self.max_lag)
        )
        X_train = pd.concat(iter_data, axis=1).dropna().to_numpy()
        return X_train, Y_train
