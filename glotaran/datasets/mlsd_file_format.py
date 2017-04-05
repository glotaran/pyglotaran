from glotaran.model import Dataset
from .spectral_timetrace import SpectralTimetrace, SpectralUnit
from enum import Enum
import os.path
import csv
import re
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
    Abstract class representing either a time- or wavelength-explicit file.
    """
    def __init__(self, file):
        self._file = file
        self._file_data_format = None
        self._observations = []  # TODO: choose name: data_points, observations, data
        self._times = []
        self._spectral_indices = []
        self._label = ""
        self._comment = ""

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

    def write(self, filename, type, overwrite=False, comment=""):
        if not isinstance(type, DataFileType):
            raise TypeError("Export type not supported")

        #self._dataset = dataset

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

        datawriter = csv.writer(f, delimiter='\t')

        datawriter.writerow(self.get_explicit_axis())

        for i in range(len(self.get_secondary_axis())):
            datawriter.writerow(self.get_data_row(i)
                                .prepend(self.get_secondary_axis()[i]))

        f.close()

    def read(self, label, spectral_unit=SpectralUnit.nm, time_unit="s"):
        if not os.path.isfile(self._file):
            raise Exception("File does not exist.")

        self._label = label
        with open(self._file) as f:
            self._file_data_format = get_data_file_format(f)  # TODO: what to do with return: None?
            if not self._file_data_format:
                ImportError

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
        times = np.array([0])
        for l in protocol:
            val = l.strip().split()
            period = float(val[1])
            cycle = float(val[2])
            print('period={}, cycle={}'.format(period,cycle))
            print(np.arange(period + times[-1], times[-1] + period + cycle * period, period))
            times = np.concatenate((times, np.arange(period + times[-1], times[-1] + period + cycle * period, period)))
        print('len(times)={}'.format(len(times)))

        self._times = times

    def read_data(self, f):
        f.seek(0)
        line = f.readline()
        while line:
            if line.startswith("[") and line.startswith(HeaderMLSDMulheim.data.value):
                break
            line = f.readline()
        df = pd.read_table(f,header=None,index_col=None)
        raw_data = df.values
        self._spectral_indices = raw_data[:,0]
        data = np.empty((raw_data.shape[0],int((raw_data.shape[1]-1)/3)))
        for i,j in zip(range(0,data.shape[1],1),range(1,raw_data.shape[1],3)):
            data[:,i] = raw_data[:,j+2] - (raw_data[:,j]+raw_data[:,j+1])/2
        self._observations = data
        #self._times = range(0,data.shape[1])


def get_data_file_format(f):
    required_headers = ("[number, period, cycle, actinic light]","[data, wavelength(once), 1st dark, 2nd dark, measure]")
    nmatches=0
    data_file_format = None
    line = ""
    while not data_file_format:
        line = f.readline()
        if line.startswith("["):
            if any(substring in line for substring in required_headers):
                nmatches += 1
                if nmatches > 1 :
                    data_file_format=DataFileType.mlsd_mulheim
                    break

    return data_file_format
