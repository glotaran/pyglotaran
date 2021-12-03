import xarray as xr

from glotaran.testing.parallel_spectral_decay import DATASET as parallel_dataset
from glotaran.testing.sequential_spectral_decay import DATASET as sequential_dataset


def test_dataset():
    assert isinstance(parallel_dataset, xr.Dataset)
    assert isinstance(sequential_dataset, xr.Dataset)
