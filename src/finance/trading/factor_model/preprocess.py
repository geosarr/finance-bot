from dataclasses import dataclass
import pandas as pd


@dataclass
class PreProcess:
    max_lag: int = 50

    def split_x_y(self, price: pd.DataFrame):
        df = price.pct_change().dropna().reset_index(drop=True).T
        Y_train = df.loc[:, self.max_lag :]
        iter_data = map(
            lambda pos: df.T.shift(pos + 1).stack(dropna=False), range(self.max_lag)
        )
        X_train = pd.concat(iter_data, axis=1).dropna()
        return X_train, Y_train
