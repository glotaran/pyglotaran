from __future__ import annotations

import xarray as xr

from glotaran.io import Io
from glotaran.io import register_io
from glotaran.project import SavingOptions
from glotaran.project import default_data_filters


@register_io("nc")
class NetCDFIo(Io):
    @staticmethod
    def read_dataset(fmt: str, file_name: str) -> xr.Dataset | xr.DataArray:
        return xr.open_dataset(file_name)

    @staticmethod
    def write_dataset(
        fmt: str, file_name: str, saving_options: SavingOptions, dataset: xr.Dataset
    ):

        data_to_save = dataset

        data_filter = (
            saving_options.data_filter
            if saving_options.data_filter is not None
            else default_data_filters[saving_options.level]
        )

        if data_filter is not None:

            data_to_save = xr.Dataset()
            for item in data_filter:
                data_to_save[item] = dataset[item]

        data_to_save.to_netcdf(file_name)
