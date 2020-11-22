import pathlib

import xarray as xr

known_reading_formats = {}


def read_data_file(filename: str, fmt: str = None) -> xr.Dataset:
    path = pathlib.Path(filename)

    if fmt is None:
        fmt = path.suffix[1:] if path.suffix != "" else "nc"

    if fmt not in known_reading_formats:
        raise Exception(
            f"Unknown file format '{fmt}'."
            f"Supported formats are {list(known_reading_formats.keys())}"
        )

    return known_reading_formats[fmt].read(filename)


class Reader:
    def __init__(self, extension, name, reader):
        self.extension = extension
        self.name = name or f" '.{extension}' format"
        self.read = reader


def file_reader(extension: str = None, name: str = None):
    def decorator(reader):
        known_reading_formats[extension] = Reader(extension, name, reader)
        return reader

    return decorator


file_reader(extension="nc", name="netCDF4")(xr.open_dataset)
