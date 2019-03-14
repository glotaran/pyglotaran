import pathlib
import xarray as xr

from .wavelength_time_explicit_file import read_ascii_time_trace
from .sdt_file_reader import read_sdt

known_formats = {
    'nc': xr.open_dataset,
    'ascii': read_ascii_time_trace,
    'sdt': read_sdt,
}


def read_data_file(filename: str, fmt: str = None) -> xr.Dataset:
    path = pathlib.Path(filename)

    if fmt is None:
        fmt = path.suffix[1:] if path.suffix != '' else 'nc'

    if fmt not in known_formats:
        raise Exception(
            f"Unknown filne format '{fmt}'. Supported formats are [{known_formats.keys()}]")

    return known_formats[fmt](filename)
