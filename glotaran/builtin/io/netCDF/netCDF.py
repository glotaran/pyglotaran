from __future__ import annotations

import xarray as xr

from glotaran.io import DataIoInterface
from glotaran.io import register_data_io
from glotaran.project import SavingOptions
from glotaran.project import default_data_filters


@register_data_io("nc")
class NetCDFDataIo(DataIoInterface):
    def load_dataset(self, file_name: str) -> xr.Dataset | xr.DataArray:
        return xr.open_dataset(file_name)

    def save_dataset(
        self,
        dataset: xr.Dataset,
        file_name: str,
        *,
        saving_options: SavingOptions = SavingOptions(),
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
