import xarray as xr

from glotaran.examples.sequential import dataset


def test_dataset():
    assert isinstance(dataset, xr.Dataset)
