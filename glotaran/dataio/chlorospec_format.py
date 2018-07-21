from glotaran.models.spectral_temporal.dataset import SpectralTemporalDataset
from natsort import natsorted
from glob import glob
import os
import numpy as np


def load_samples(samples_path):
    with open(samples_path, 'rb') as f:
        shape = get_header(f, 2)
        data = get_data(f, shape)
        return data


def load_times(times_path):
    with open(times_path, 'rb') as f:
        shape = get_header(f, 1)
        data = get_data(f, shape)
        return data


def get_header(f, dims):
    header = np.fromfile(f, dtype=">i", count=dims)
    return header


def get_data(f, shape):
    data = np.fromfile(f, dtype=">d")
    return data.reshape(shape).astype(np.float)


class ChlorospecData(object):
    """
    Class capable of reading in binary data in the structure
    FOLDER/NEXPERIMENT/NREPEAT/DATA
    Returns a glotaran.model.dataset
    """

    def __init__(self, folder, debug=False):
        self._data_folder = folder
        self._label = ""

    def read(self, label):
        if not os.path.isdir(self._data_folder):
            raise Exception("Data folder does not exist.")
        if not ChlorospecData.is_valid_path(self._data_folder):
            raise Exception("Not a valid data folder")

        # with open(self._data_folder) as f:
        # Recognize binary signature?
        # self._file_data_format = get_data_file_format(f)  # TODO: what to do with return: None?
        # if not self._file_data_format:
        #    ImportError

        self._label = label
        f = self._data_folder
        sorted_sub_folders = ChlorospecData.valid_sub_folders_natural_sorted(f)
        t0 = 0
        all_times = np.empty(shape=0)
        data = np.empty(shape=(0, 0))

        for idx, s in enumerate(sorted_sub_folders):
            sf = os.path.join(f, s)
            times_path = os.path.join(sf, "times.bin")
            samples_path = os.path.join(sf, "spectra.bin")
            if os.path.isfile(times_path) and os.path.isfile(samples_path):
                times = load_times(times_path)
                if idx == 0:
                    t0 = times[0]
                times = [-t0 + t for t in times]
                all_times = np.concatenate((all_times, times))
                samples = load_samples(samples_path)
                if data.size == 0:
                    data = samples.T  # our dataset convention
                else:
                    data = np.concatenate((data, samples.T), 1)
        wavelengths = np.linspace(159.735, 1047.157, data.shape[0])
        dataset = SpectralTemporalDataset(self._label)
        dataset.set_axis("time", all_times)
        dataset.set_axis("spectral", wavelengths)
        dataset.data = data
        return dataset

    @staticmethod
    def valid_sub_folders_natural_sorted(path):
        sub_folders = [f.path for f in os.scandir(path) if f.is_dir() and
                       glob(os.path.join(f.path, '*.bin'))]
        return natsorted(sub_folders, key=lambda x: x.lower())

    @staticmethod
    def is_valid_path(path):
        return os.path.isdir(path) and \
               len(ChlorospecData.valid_sub_folders_natural_sorted(path)) > 0
