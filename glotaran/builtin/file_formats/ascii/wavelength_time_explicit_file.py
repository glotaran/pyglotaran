from enum import Enum
import os.path
import csv
import re
import numpy as np
import xarray as xr

from glotaran.io.reader import file_reader
from glotaran.io.prepare_dataset import prepare_time_trace_dataset


class DataFileType(Enum):
    time_explicit = "Time explicit"
    wavelength_explicit = "Wavelength explicit"


class ExplicitFile(object):
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
            self._times = np.array(dataset.coords['time'])
            self._spectral_indices = np.array(dataset.coords['spectral'])
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

    def write_old(self, filename, export_type, overwrite=False, comment=""):
        if not isinstance(type, DataFileType):
            raise TypeError("Export type not supported")

        # self._dataset = dataset

        comment = comment.splitlines()
        while len(comment) < 2:
            comment.append("")

        if os.path.isfile(filename) and not overwrite:
            raise Exception("File already exist.")

        f = open(filename, "w")

        f.write(comment[0])
        f.write(comment[1])

        f.write(self.get_format_name())

        f.write("Intervalnr {}".format(len(self.get_explicit_axis())))

        datawriter = csv.writer(f, delimiter=r'\t')

        datawriter.writerow(self.get_explicit_axis())

        for i in range(len(self.get_secondary_axis())):
            datawriter.writerow(self.get_data_row(i)
                                .prepend(self.get_secondary_axis()[i]))

        f.close()

    def write(self, overwrite=False, comment="",
              file_format="TimeExplicit", number_format="%.10e"):
        # TODO: write a more elegant method

        if os.path.isfile(self._file) and not overwrite:
            print('File {} already exists'.format(os.path.isfile(self._file)))
            raise Exception("File already exist.")

        comments = "# Filename: " + self._file + "\n" + " ".join(self._comment.splitlines()) + "\n"

        if file_format == "WavelengthExplicit":
            wav = '\t'.join([repr(num) for num in self._spectral_indices])
            header = comments + "Wavelength explicit\nIntervalnr {}" \
                                "".format(len(self._spectral_indices)) + "\n" + wav
            raw_data = np.vstack((self._times.T, self._observations)).T
        elif file_format == "TimeExplicit":
            tim = '\t'.join([repr(num) for num in self._times])
            header = comments + "Time explicit\nIntervalnr {}" \
                                "".format(len(self._times)) + "\n" + tim
            raw_data = np.vstack((self._spectral_indices.T, self._observations.T)).T
        else:
            raise NotImplementedError

        np.savetxt(self._file, raw_data, fmt=number_format, delimiter='\t', newline='\n',
                   header=header, footer='', comments='')

    def read(self, prepare: bool = True):
        if not os.path.isfile(self._file):
            raise Exception("File does not exist.")

        with open(self._file) as f:
            f.readline()  # Read first line with comments (and discard for now)
            f.readline()  # Read second line with comments (and discard for now)
            # TODO: what to do with return: None?
            self._file_data_format = get_data_file_format(f.readline())
            # TODO: what to do with return: None?
            interval_nr = get_interval_number(f.readline().strip().lower())
            all_data = []
            line = f.readline()
            while line:
                all_data.append([float(i) for i in re.split(r"\s+|\t+|\s+\t+|\t+\s+|\u3000+",
                                                            line.strip())])
                # data_block = pd.read_csv(line, sep="\s+|\t+|\s+\t+|\t+\s+|\u3000+",
                #                          engine='python', header=None, index_col=False)
                # all_data.append(data_block.values())
                line = f.readline()
            all_data = np.asarray(all_data)

            interval_counter = -interval_nr
            explicit_axis = []
            secondary_axis = []
            observations = [[]]
            obs_idx = 0

            for item in [sl for sublist in all_data for sl in sublist]:
                if item != item:  # NaN was found
                    ValueError()
                if interval_counter < 0:
                    # print("explicit_axis {}: {}".format(interval_counter, item))
                    explicit_axis.append(item)
                elif interval_counter == 0:
                    # print("secondary_axis: {}".format(item))
                    secondary_axis.append(item)
                elif interval_counter < (interval_nr + 1):
                    observations[obs_idx].append(item)
                interval_counter += 1
                if interval_counter == (interval_nr + 1):
                    interval_counter = 0
                    obs_idx += 1
                    observations.append([])

            del observations[-1]

            if self._file_data_format == DataFileType.time_explicit:
                self._times = np.asarray(explicit_axis)
                self._spectral_indices = np.asarray(secondary_axis)
                self._observations = np.asarray(observations)

            elif self._file_data_format == DataFileType.wavelength_explicit:
                self._spectral_indices = np.asarray(explicit_axis)
                self._times = np.asarray(secondary_axis)
                self._observations = np.asarray(observations)

            else:
                NotImplementedError()
                pass

        return self.dataset(prepare=prepare)

    def dataset(self, prepare: bool = True):
        data = self._observations
        if self._file_data_format == DataFileType.time_explicit:
            data = data.T
        dataset = xr.DataArray(
            data, coords=[('time', self._times), ('spectral', self._spectral_indices)])
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
        row = []
        pass
        return row

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
    interval_number = None
    match = re.search(r"intervalnr\s(.*)", line.strip().lower())
    if match:
        interval_number = match.group(1)
    if not interval_number:
        interval_number = re.search(r'\d+', line[::-1]).group()[::-1]
    try:
        interval_number = int(interval_number)
    except ValueError:
        pass  # TODO: let user know no interval_number was found
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
        NotImplementedError()
    return data_file_format


@file_reader(extension='ascii', name='Wavelength-/Time-Explicit ASCII')
def read_ascii_time_trace(fname: str, prepare: bool = True) -> xr.Dataset:
    """Reads an ascii file in wavelength- or time-explicit format.

    See [1] for documentation of this format.

    Parameters
    ----------
    fname : str
        Name of the ascii file.

    Returns
    -------
    dataset : xr.Dataset

    Notes
    -----
    .. [1] http://glotaran.org/wiki/doku.php?id=file_formats
    """

    data_file_format = None
    with open(fname) as f:
        f.readline()  # Read first line with comments (and discard for now)
        f.readline()  # Read second line with comments (and discard for now)
        data_file_format = get_data_file_format(f.readline())

    data_file = WavelengthExplicitFile(filepath=fname) if data_file_format is \
        DataFileType.wavelength_explicit else TimeExplicitFile(fname)

    return data_file.read(prepare=prepare)


def write_ascii_time_trace(filename: str,
                           dataset: xr.DataArray,
                           overwrite=False,
                           comment="",
                           file_format="TimeExplicit",
                           number_format="%.10e"):
    data_file = \
        TimeExplicitFile(filepath=filename, dataset=dataset) if file_format == "TimeExplicit" \
        else WavelengthExplicitFile(filepath=filename, dataset=dataset)
    data_file.write(overwrite=overwrite,
                    comment=comment,
                    file_format=file_format,
                    number_format=number_format)
