from dataclasses import dataclass
from enum import Enum, auto
import polars as pl
import pandas as pd

from finance.constant import PARQUET


class Engine(Enum):
    POLARS = auto()
    PANDAS = auto()


@dataclass
class Loader:
    path: str
    engine: Engine

    def __post_init__(self):
        self.__path = (
            self.path.rstrip(PARQUET)
            if self.path.endswith(PARQUET) and self.engine == Engine.PANDAS
            else self.path
        )

    def load(self):
        if self.engine == Engine.POLARS:
            data = pl.scan_parquet(self.__path)
        elif self.engine == Engine.PANDAS:
            data = pd.read_parquet(self.__path)
        return data
