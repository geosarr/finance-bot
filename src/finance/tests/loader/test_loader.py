import unittest
from finance.loader import *
import numpy as np
import os
import shutil


class TestLoader(unittest.TestCase):
    @staticmethod
    def create_temp_parquet(path: str) -> None:
        path = path + "/" if not path.endswith("/") else path
        os.makedirs(path, exist_ok=True)
        init_seed = np.random.SeedSequence(0)
        seeds = init_seed.spawn(100)
        ncols = 10
        nrows = 100
        columns = [f"col{i}" for i in range(ncols)]
        for pos, seed in enumerate(seeds):
            gen = np.random.default_rng(seed)
            data = gen.normal(size=(nrows, ncols))
            data = pd.DataFrame(data, columns=columns)
            data.to_parquet(f"{path}{pos}.parquet")

    def setUp(self) -> None:
        temp_dir = "./test/"
        self.create_temp_parquet(temp_dir)
        self.attr = {"path": f"{temp_dir}*.parquet", "engine": Engine.PANDAS}
        self.model = Loader(*self.attr)
        shutil.rmtree(temp_dir)

    def test_loader(self) -> None:
        pass
