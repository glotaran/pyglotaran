from __future__ import annotations

import os.path
import re
import warnings
from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr

from glotaran.io import DataIoInterface
from glotaran.io import register_data_io
from glotaran.io.prepare_dataset import prepare_time_trace_dataset

#  from glotaran.io.reader import file_reader


class DataFileType(Enum):
    time_explicit = "Time explicit"
    wavelength_explicit = "Wavelength explicit"


class ExplicitFile:
    """
    Abstract class representing either a time- or wavelength-explicit file.
    """

    # TODO: implement time_intervals
    def __init__(self, filepath: str = None, dataset: xr.DataArray = None):
        self._file_data_format = None
        self._observations = []  # TODO: choose name: data_points, observations, data
        self._times = []
        self._spectral_indices = []
        self._label = ""
        self._comment = ""
        absfilepath = os.path.realpath(filepath)
        if dataset is not None:
            self._observations = np.array(dataset.values).T
            self._times = np.array(dataset.coords["time"])
            self._spectral_indices = np.array(dataset.coords["spectral"])
            self._file = filepath
        elif os.path.isfile(filepath):
            self._file = filepath
        elif os.path.isfile(absfilepath):
            self._file = absfilepath
        else:
            raise Exception(f"Path does not exist: {filepath}, {absfilepath}")

    def get_explicit_axis(self):
        raise NotImplementedError

    def set_explicit_axis(self, axis):
        raise NotImplementedError

    def get_secondary_axis(self):
        raise NotImplementedError

    def get_data_row(self, index):
        raise NotImplementedError

    def get_observations(self, index):
        raise NotImplementedError

    def get_format_name(self):
        raise NotImplementedError

    def write(
        self,
        overwrite=False,
        comment="",
        file_format=DataFileType.time_explicit,
        number_format="%.10e",
    ):
        # TODO: write a more elegant method

        if os.path.isfile(self._file) and not overwrite:
            raise FileExistsError(f"File already exist:\n{self._file}")
        comment = self._comment + " " + comment

        comments = "# Filename: " + str(self._file) + "\n" + " ".join(comment.splitlines()) + "\n"

        if file_format == DataFileType.wavelength_explicit:
            wav = "\t".join(repr(num) for num in self._spectral_indices)
            header = (
                comments + "Wavelength explicit\nIntervalnr {}"
                "".format(len(self._spectral_indices)) + "\n" + wav
            )
            raw_data = np.vstack((self._times.T, self._observations)).T
        elif file_format == DataFileType.time_explicit:
            tim = "\t".join(repr(num) for num in self._times)
            header = (
                comments + "Time explicit\nIntervalnr {}" "".format(len(self._times)) + "\n" + tim
            )
            raw_data = np.vstack((self._spectral_indices.T, self._observations.T)).T
        else:
            raise NotImplementedError

        np.savetxt(
            self._file,
            raw_data,
            fmt=number_format,
            delimiter="\t",
            newline="\n",
            header=header,
            footer="",
            comments="",
        )

    def read(self, prepare: bool = True):
        if not os.path.isfile(self._file):
            raise Exception("File does not exist.")
        with open(self._file) as f:
            f.readline()  # The first two lines are comments
            f.readline()
            # The third line defines the ExplicitFileFormat (Time or Wavelength explicit)
            self._file_data_format = get_data_file_format(f.readline())
            # The fourth line define the number of elements on the explicit axis, which
            # we can ignore because pandas is intelligent enough to read it
        # read the first line (explicit_axis) separately
        explicit_axis = pd.read_csv(
            self._file, skiprows=4, delimiter=r"\s+", header=None, nrows=1
        ).values
        explicit_axis = explicit_axis[0, :]  # reshape to (n,)
        # then the rest of the data:
        rest_of_data = pd.read_csv(self._file, skiprows=5, delimiter=r"\s+", header=None).values
        secondary_axis = rest_of_data[:, 0]
        observations = rest_of_data[:, 1:]
        if self._file_data_format == DataFileType.time_explicit:
            self._times = explicit_axis  # (501,)
            self._spectral_indices = secondary_axis  # (51,)
            self._observations = observations  # len(observation)=51 . (51, 501)
        elif self._file_data_format == DataFileType.wavelength_explicit:
            self._spectral_indices = explicit_axis
            self._times = secondary_axis
            self._observations = observations
        else:
            raise NotImplementedError()
        return self.dataset(prepare=prepare)

    def dataset(self, prepare: bool = True) -> xr.Dataset | xr.DataArray:
        data = self._observations
        if self._file_data_format == DataFileType.time_explicit:
            data = data.T
        dataset = xr.DataArray(
            data, coords=[("time", self._times), ("spectral", self._spectral_indices)]
        )
        if prepare:
            dataset = prepare_time_trace_dataset(dataset)
        return dataset


class WavelengthExplicitFile(ExplicitFile):
    """
    Represents a wavelength explicit file
    """

    def get_explicit_axis(self):
        return self._spectral_indices

    def get_secondary_axis(self):
        return self.observations()

    def get_data_row(self, index):
        return []

    def add_data_row(self, row):
        if self._timepoints is None:
            self._timepoints = []
        self._timepoints.append(float(row.pop(0)))

        if self._spectra is None:
            self._spectra = []
        self._spectra.append(float(row))

    def get_format_name(self):
        return DataFileType.wavelength_explicit

    def times(self):
        return self.get_secondary_axis()

    def wavelengths(self):
        return self.get_explicit_axis()


class TimeExplicitFile(ExplicitFile):
    """
    Represents a time explicit file
    """

    def get_explicit_axis(self):
        return self.observations()

    def set_explicit_axis(self, axies):
        self._timepoints = float(axies)

    def get_secondary_axis(self):
        return self.channel_labels

    def get_data_row(self, index):
        return self.get_channel(self.channel_labels()[index])

    def add_data_row(self, row):
        if self._spectral_indices is None:
            self._spectral_indices = []
        self._spectral_indices.append(row.pop(0))

        if self._spectra is None:
            self._spectra = []
        self._spectra.append(float(row))

    def get_format_name(self):
        return DataFileType.time_explicit


def get_interval_number(line):
    match = re.search(r"intervalnr\s(.*)", line.strip().lower())
    interval_number = match.group(1) if match else None
    if not interval_number:
        interval_number = re.search(r"\d+", line[::-1]).group()[::-1]
    try:
        interval_number = int(interval_number)
    except ValueError:
        warnings.warn(f"No interval number found in line:\n{line}")
        interval_number = None
    return interval_number


def get_data_file_format(line):
    data_file_format = None
    if re.search(r"time\s+explicit|time\t+explicit", line.strip().lower()):
        # print("Time explicit format") #TODO: verbosity / debug statement
        data_file_format = DataFileType.time_explicit
    elif re.search(r"wavelength\s+explicit|wavelength\t+explicit", line.strip().lower()):
        # print("Wavelength explicit format") #TODO: verbosity / debug statement
        data_file_format = DataFileType.wavelength_explicit
    else:
        raise NotImplementedError()
    return data_file_format


#  @file_reader(extension="ascii", name="Wavelength-/Time-Explicit ASCII")
@register_data_io("ascii")
class AsciiDataIo(DataIoInterface):
    def load_dataset(self, file_name: str) -> xr.Dataset | xr.DataArray:
        """Reads an ascii file in wavelength- or time-explicit format.

        See [1]_ for documentation of this format.

        Parameters
        ----------
        fname : str
            Name of the ascii file.

        Returns
        -------
        dataset : xr.Dataset

        Notes
        -----
        .. [1] https://glotaran.github.io/legacy/file_formats
        """

        with open(file_name) as f:
            f.readline()  # Read first line with comments (and discard for now)
            f.readline()  # Read second line with comments (and discard for now)
            data_file_format = get_data_file_format(f.readline())

        data_file = (
            WavelengthExplicitFile(filepath=file_name)
            if data_file_format is DataFileType.wavelength_explicit
            else TimeExplicitFile(file_name)
        )

        return data_file.read(prepare=True)

    def save_dataset(
        self,
        dataset: xr.DataArray,
        file_name: str,
        *,
        comment: str = "",
        file_format: DataFileType = DataFileType.time_explicit,
        number_format: str = "%.10e",
    ):
        data_file = (
            TimeExplicitFile(filepath=file_name, dataset=dataset)
            if file_format is DataFileType.time_explicit
            else WavelengthExplicitFile(filepath=file_name, dataset=dataset)
        )
        data_file.write(
            overwrite=True, comment=comment, file_format=file_format, number_format=number_format
        )
