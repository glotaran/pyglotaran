"""Functions for data IO"""

from . import (
    prepare_dataset,
    wavelength_time_explicit_file,
    sdt_file_reader
)

prepare_dataset = prepare_dataset.prepare_dataset

read_ascii_time_trace = wavelength_time_explicit_file.read_ascii_time_trace
write_ascii_time_trace = wavelength_time_explicit_file.write_ascii_time_trace

read_sdt_data = sdt_file_reader.read_sdt
