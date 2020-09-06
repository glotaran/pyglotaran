"""
Glotarans module to read files
"""
import warnings

import numpy as np
import xarray as xr
from sdtfile import SdtFile

from glotaran.io.prepare_dataset import prepare_time_trace_dataset
from glotaran.io.reader import file_reader


@file_reader(extension="sdt", name="Becker & Hickel SDT file format.")
def read_sdt(
    file_path: str,
    index: np.ndarray = None,
    flim: bool = False,
    dataset_index: int = None,
    swap_axis: bool = False,
    orig_time_axis_index: int = 2,
) -> xr.Dataset:
    """
    Reads a `*.sdt` file and returns a pd.DataFrame (`return_dataframe==True`), a
    SpectralTemporalDataset (`type_of_data=='st'`) or a FLIMDataset (`type_of_data=='flim'`).

    Parameters
    ----------
    file_path: str
        Path to the sdt file which should be read.

    index: list, np.ndarray
        This is only needed if `type_of_data=="st"`, since `*.sdt` files,
        which only contain spectral temporal data, lack the spectral information.
        Thus for the spectral axis data need to be given by the user.

    flim:
        Set true if reading a result from a FLIM measurement.

    dataset_index: int: default 0
        If the `*.sdt` file contains multiple datasets the index will used
        to select the wanted one

    swap_axis: bool, default False
        Flag to switch a wavelength explicit `input_df` to time explicit `input_df`,
        before generating the SpectralTemporalDataset.

    orig_time_axis_index: int
        Index of the axis which corresponds to the time axis.
        I.e. for data of shape (64, 64, 256), which are a 64x64 pixel map
        with 256 time steps, orig_time_axis_index=2.

    Raises
    ______
    IndexError:
        If the length of the index array is incompatible with the data.
    """
    sdt_parser = SdtFile(file_path)
    if not dataset_index:
        # looking at the source code of SdtFile, times and data
        # always have the same len, so only one needs to be checked
        nr_of_datasets = len(sdt_parser.times)
        if nr_of_datasets > 1:
            warnings.warn(
                UserWarning(
                    f"The file '{file_path}' contains {nr_of_datasets} Datasets.\n "
                    f"By default only the first Dataset will be read. "
                    f"If you only need the first Dataset and want get rid of "
                    f"this warning you can set dataset_index=0."
                )
            )
        dataset_index = 0
    times = sdt_parser.times[dataset_index]
    data = sdt_parser.data[dataset_index]

    if index and len(index) is not data.shape[0]:
        raise IndexError(
            f"The Dataset contains {data.shape[0]} measurements, but the "
            f"indices supplied are {len(index)}."
        )
    elif not index and not flim:
        warnings.warn(
            UserWarning(
                f"There was no `index` provided."
                f"That for the indices will be a entry count(integers)."
                f"To prevent this warning from being shown, provide "
                f"a list of indices, with len(index)={data.shape[0]}"
            )
        )

    if flim:

        if orig_time_axis_index != 2:
            np.swapaxes(data, 2, orig_time_axis_index)

        full_data = xr.DataArray(data, coords={"time": times}, dims=["x", "y", "time"])
        data = full_data.stack(pixel=("x", "y")).to_dataset(name="data")
        data["full_data"] = full_data.rename({"x": "pixel_x", "y": "pixel_y"})
        data["data_intensity_map"] = (
            data.data.groupby("pixel").sum().unstack().rename({"x": "pixel_x", "y": "pixel_y"})
        )
    else:
        if swap_axis:
            data = data.T
        if not index:
            index = np.array(range(data[0]))
        data = xr.DataArray(data.T, coords=[("time", times), ("spectral", index)])
        data = prepare_time_trace_dataset(data)
    return data
