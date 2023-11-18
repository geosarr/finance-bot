import unittest
from finance.loader import *
import numpy as np
import os
import shutil


class TestLoader(unittest.TestCase):
    @staticmethod
    def create_temp_parquet(path: str, ncols: int, nrows: int, nfiles: int) -> None:
        path = path + "/" if not path.endswith("/") else path
        os.makedirs(path, exist_ok=True)
        init_seed = np.random.SeedSequence(0)
        seeds = init_seed.spawn(nfiles)
        columns = [f"col{i}" for i in range(ncols)]
        for pos, seed in enumerate(seeds):
            gen = np.random.default_rng(seed)
            data = gen.normal(size=(nrows, ncols))
            data = pd.DataFrame(data, columns=columns)
            data.to_parquet(f"{path}{pos}.parquet")

    def setUp(self) -> None:
        temp_dir = "./test/"
        self.ncols = 10
        self.nrows = 100
        self.nfiles = 100
        self.create_temp_parquet(temp_dir, self.ncols, self.nrows, self.nfiles)
        self.attr_pd = {"path": f"{temp_dir}", "engine": Engine.PANDAS}
        self.attr_pl = {"path": f"{temp_dir}/*.parquet", "engine": Engine.POLARS}
        self.loader_pd = Loader(**self.attr_pd)
        self.loader_pl = Loader(**self.attr_pl)
        self.data_pd = self.loader_pd.load()
        self.data_pl = self.loader_pl.load()
        shutil.rmtree(temp_dir)

    def test_loader(self) -> None:
        self.assertEqual(self.data_pd.shape, (self.nrows * self.nfiles, self.ncols))
        self.assertEqual(self.data_pl.width, self.ncols)
