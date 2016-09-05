from glotaran_core import Dataset, Datasets, Model, GlobalAnalysis
import numpy as np
from unittest import TestCase

kinpar = np.asarray([.006667, .006667, 0.00333, 0.00035, 0.0303, 0.000909])


class TestData(Dataset):
    def __init__(self):
        super(Dataset, "testdata")
        self._times = np.asarray(np.arange(0, 1500, 1.5))
        self._wavenum = np.asarray(np.arange(12820, 15120, 4.6))
        location = np.asarray([14705, 13513, 14492, 14388, 14184, 13986])
        self._data = np.empty((self._wavenum.size, location.size),
                              dtype=np.float64, order="F")

        delta = np.asarray([400, 1000, 300, 200, 350, 330])
        amp = np.asarray([1, 0.2, 1, 1, 1, 1])
        for i in range(location.size):
            self._data[:, i] = {amp[i] * np.exp(-np.log(2) *
                                np.square(2 *
                                (self._wavenum - location[i])/delta[i]))}

    def wavenumbers(self):
        return self._wavenum

    def timepoints(self):
        return self._times

    def data(self):
        return self._data


class TestModel(Model):
    def __init__(self):
        pass


class TestSimpleSerial(TestCase):
    def test_simple_serial(self):
        data = Datasets()
        data.add(TestData())
        model = TestModel()
        analysis = GlobalAnalysis(data, model)

        analysis.fit()

        epsilon_percent = 0.1
        for i in range(analysis.result().best_fit_parameters()):
            real_par = kinpar[i]
            epsilon = real_par * epsilon_percent
            fitted_par = analysis.result().best_fit_parameters()

            self.assertTrue(fitted_par > real_par - epsilon and
                            fitted_par < real_par + epsilon)
