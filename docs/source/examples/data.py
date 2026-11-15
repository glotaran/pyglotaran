# /// script
# requires-python = ">=3.13"
# dependencies = ["pyglotaran>=0.8"]
# ///
"""Create a dataset from arrays or a table, add a weight, and save and load it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from glotaran.io import load_dataset
from glotaran.io import save_dataset

time = np.linspace(0, 20, 201)
spectral = np.linspace(600, 700, 31)
signal = np.outer(np.exp(-0.6 * time), np.exp(-(((spectral - 630) / 15) ** 2)))

# From arrays: a variable `data` with the dimensions `time` and `spectral`.
dataset = xr.Dataset(
    {"data": (("time", "spectral"), signal)},
    coords={"time": time, "spectral": spectral},
)
print("from arrays:", dict(dataset.sizes), list(dataset.data_vars))

# From a table with one row for each time and one column for each wavelength.
table = pd.DataFrame(signal, index=time, columns=spectral)
from_table = xr.Dataset(
    {"data": (("time", "spectral"), table.to_numpy())},
    coords={"time": table.index.to_numpy(), "spectral": table.columns.to_numpy(dtype=float)},
)
print("from a table:", dict(from_table.sizes))

# A weight has the shape of the data. Here the first 5 wavelengths count ten times less.
dataset["weight"] = xr.ones_like(dataset.data)
dataset["weight"].loc[{"spectral": slice(600, 615)}] = 0.1
print("weights:", np.unique(dataset.weight).tolist())

# NetCDF keeps the dataset as it is, including the weight.
save_dataset(dataset, "data_example.nc", allow_overwrite=True)
loaded = load_dataset("data_example.nc")
print("from data_example.nc:", dict(loaded.sizes), sorted(loaded.data_vars))

# The ascii format holds one array, so it takes the data without the weight.
save_dataset(dataset.data, "data_example.ascii", allow_overwrite=True)
loaded = load_dataset("data_example.ascii")
print("from data_example.ascii:", dict(loaded.sizes), sorted(loaded.data_vars))
print("same values:", bool(np.allclose(loaded.data.transpose("time", "spectral"), signal)))
