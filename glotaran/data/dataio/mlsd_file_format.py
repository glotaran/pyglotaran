from glotaran.model import Dataset
# from .spectral_timetrace import SpectralTimetrace, SpectralUnit
from enum import Enum
import os.path
import numpy as np
import pandas as pd


class DataFileType(Enum):
    mlsd_mulheim = "MLSD Mulheim"


class HeaderMLSDMulheim(Enum):
    int_time = "[integration time:]"
    int_delay = "[integration delay:]"
    boxcar = "[boxcar width:]"
    avg = "[average:]"
    dark = "[dark spectrum:]"
    comment = "[comment:]"
    prot = "[number, period, cycle, actinic light]"
    data = "[data, wavelength(once), 1st dark, 2nd dark, measure]"


class MLSDFile(object):
    """
    Class capable of reading in so called time- or wavelength-explicit file format.
    Returns a glotaran.model.dataset
    """
    def __init__(self, file, debug=False):
        self._file = file
        self._file_data_format = None
        self._observations = []  # TODO: choose name: data_points, observations, data
        self._times = []
        self._spectral_indices = []
        self._label = ""
        self._comment = ""
        self._debug = debug

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

    def write(self, filename, overwrite=False, comment="", file_format="Time explicit"):
        # TODO: write a more elegant method

        if os.path.isfile(filename) and not overwrite:
            print('File {} already exists'.format(os.path.isfile(filename)))
            raise Exception("File already exist.")

        comments = "# Filename: " + self._file + "\n" + " ".join(self._comment.splitlines()) + "\n"

        if file_format == "Wavelength explicit":
            wav = '\t'.join([repr(num) for num in self._spectral_indices])
            header = comments + "Wavelength explicit\nIntervalnr {}" \
                                "".format(len(self._spectral_indices)) + "\n" + wav
            raw_data = np.vstack((self._times.T, self._observations)).T
        elif file_format == "Time explicit":
            tim = '\t'.join([repr(num) for num in self._times])
            header = comments + "Time explicit\nIntervalnr {}" \
                                "".format(len(self._times)) + "\n" + tim
            raw_data = np.vstack((self._spectral_indices.T, self._observations.T)).T
        else:
            raise NotImplementedError

        np.savetxt(filename, raw_data, fmt='%.18e', delimiter='\t', newline='\n',
                   header=header, footer='', comments='')

    def read(self, label):
        if not os.path.isfile(self._file):
            raise Exception("File does not exist.")

        self._label = label
        with open(self._file) as f:
            self._file_data_format = get_data_file_format(f)  # TODO: what to do with return: None?
            if not self._file_data_format:
                raise ImportError

            self.read_comment(f)
            self.read_prot(f)
            self.read_data(f)

    def dataset(self):
        dataset = Dataset(self._label)
        if not self._file_data_format:
            return None
        elif self._file_data_format == DataFileType.mlsd_mulheim:
            dataset.set_axis("spec", self._spectral_indices)
            dataset.set_axis("time", self._times)
        dataset.data = self._observations
        return dataset

    def read_comment(self, f):
        f.seek(0)
        comment = []
        for line in f:
            if line.startswith("[") and line.startswith(str(HeaderMLSDMulheim.comment)):
                break
        for line in f:
            if line.startswith("["):
                break
            elif not line.strip():
                pass
            else:
                comment.append(line)
        self._comment = self._file + '\n' + '_'.join(comment)

    def read_prot(self, f):
        f.seek(0)
        protocol = []
        for line in f:
            if line.startswith("[") and line.startswith(HeaderMLSDMulheim.prot.value):
                break
        for line in f:
            if line.startswith("["):
                break
            elif not line.strip():
                pass
            else:
                protocol.append(line)
        tmp = [0]
        for l in protocol:
            val = l.strip().split()
            period = float(val[1])
            cycle = float(val[2])
            for i in range(int(cycle)):
                tmp.append(tmp[-1]+period)
            times = np.array(tmp)
        if self._debug:
            print('len(times)={}'.format(len(times)))

        self._times = times

    def read_data(self, f):
        f.seek(0)
        line = f.readline()
        while line:
            if line.startswith("[") and line.startswith(HeaderMLSDMulheim.data.value):
                break
            line = f.readline()
        df = pd.read_table(f, header=None, index_col=None)
        raw_data = df.values
        self._spectral_indices = raw_data[:, 0]
        data = np.empty((raw_data.shape[0], int((raw_data.shape[1]-1)/3)))
        for i, j in zip(range(0, data.shape[1], 1), range(1, raw_data.shape[1], 3)):
            data[:, i] = raw_data[:, j+2] - (raw_data[:, j]+raw_data[:, j+1])/2
        self._observations = data
        # self._times = range(0,data.shape[1])


def get_data_file_format(f):
    required_headers = ("[number, period, cycle, actinic light]",
                        "[data, wavelength(once), 1st dark, 2nd dark, measure]")
    nmatches = 0
    data_file_format = None
    line = ""
    while not data_file_format:
        line = f.readline()
        if line.startswith("["):
            if any(substring in line for substring in required_headers):
                nmatches += 1
                if nmatches > 1:
                    data_file_format = DataFileType.mlsd_mulheim
                    break

    return data_file_format


# def is_blank (mystr):
#     return not (mystr and mystr.strip())


# def is_not_blank (mystr):
#     return bool(mystr and mystr.strip())
