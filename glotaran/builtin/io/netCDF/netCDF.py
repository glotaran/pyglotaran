from __future__ import annotations

import xarray as xr

from glotaran.io import DataIoInterface
from glotaran.io import register_data_io


@register_data_io("nc")
class NetCDFDataIo(DataIoInterface):
    def load_dataset(self, file_name: str) -> xr.Dataset | xr.DataArray:
        with xr.open_dataset(file_name) as ds:
            return ds.load()

    def save_dataset(
        self,
        dataset: xr.Dataset,
        file_name: str,
        *,
        data_filters: list[str] | None = None,
    ):

        data_to_save = dataset if data_filters is None else dataset[data_filters]
        data_to_save.to_netcdf(file_name, mode="w")
