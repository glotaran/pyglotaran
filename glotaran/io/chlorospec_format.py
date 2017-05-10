from glotaran.models.spectral_temporal.dataset import SpectralTemporalDataset
from os.path import join, isdir
from math import floor
import numpy as np

class ChlorospecData(object):
    """
    Class capable of reading in binary data in the structure
    FOLDER/NEXPERIMENT/NREPEAT/DATA
    Returns a glotaran.model.dataset
    """

    def __init__(self, folder, debug = False):
        self._data_folder = folder
        self._label = ""

    def read(self, label):
        if not isdir(self._data_folder):
            raise Exception("Data folder does not exist.")

        self._label = label
        f = self._data_folder
        # with open(self._data_folder) as f:
            # Recognize binary signature?
            # self._file_data_format = get_data_file_format(f)  # TODO: what to do with return: None?
            # if not self._file_data_format:
            #    ImportError
        samples = self.load_samples(f)
        data = samples.T #our dataset convention
        s = samples.shape
        print('spectra [5x5: {}'.format(samples[floor(s[1]/2)-3:floor(s[1]/2)+2,floor(s[1]/2)-3:floor(s[1]/2)+2]))
        times = self.load_times(f)
        print('times: {}'.format(times))
        dataset = SpectralTemporalDataset(self._label)
        dataset.set_axis("time", times)
        dataset.set_axis("spectral", np.linspace(159.735, 1047.157, data.shape[0]))
        dataset.data = data
        return dataset

    def load_samples(self, folder):
        with open(join(folder, "spectra.bin"), 'rb') as f:
            shape = self.get_header(f, 2)
            data = self.get_data(f, shape)
            return data

    def load_times(self, folder):
        with open(join(folder, "times.bin"), 'rb') as f:
            shape = self.get_header(f, 1)
            data = self.get_data(f, shape)
            return data

    def get_header(self, f, dims):
        header = np.fromfile(f, dtype=">i", count=dims)
        return header

    def get_data(self, f, shape):
        data = np.fromfile(f, dtype=">d")
        return data.reshape(shape).astype(np.float)


