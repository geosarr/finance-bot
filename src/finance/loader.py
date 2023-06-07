from dataclasses import dataclass
from enum import Enum, auto
import polars as pl
import pandas as pd

class Engine(Enum):
    POLAR = auto()
    PANDAS = auto()


@dataclass
class Loader:
    path: str
    engine: Engine
    # data: Union[pd.DataFrame, pl.LazyFrame] = None

    def load(self):
        if self.engine == Engine.POLAR:
            data = pl.scan_parquet(self.path)
        elif self.engine == Engine.PANDAS:
            data = pd.read_parquet(self.path)
        return data
            
        