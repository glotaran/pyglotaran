from glotaran_core import (Datasets, KineticModel, KineticParameter,
                           GaussianIrf, GlobalAnalysis)
from glotaran_tools import SimulatedSpectralTimetrace
from unittest import TestCase

rates = [.006667, .006667, 0.00333, 0.00035, 0.0303, 0.000909]
amplitudes = [3, 6, 8, 1, 2.5, 4]
positions = [150, 300, 350, 420, 560, 600]
widths = [20, 20, 20, 20, 20, 20]


class TestSimpleSerial(TestCase):
    def test_simple_serial(self):
        data = Datasets()
        data.add(SimulatedSpectralTimetrace(amplitudes, rates, positions,
                                            widths, 100, 700, 1, 5e-9, 1e-10,
                                            label="TestData"))
        kinpar = []
        for i in range(len(rates)):
            kinpar.append(KineticParameter("k{}".format(i), rates[i], i))
        irf = GaussianIrf(450, 100)
        analysis = GlobalAnalysis(data, KineticModel("TestData", kinpar, irf))

        analysis.fit()

        epsilon_percent = 0.1
        for i in range(analysis.result().best_fit_parameters()):
            real_par = kinpar[i]
            epsilon = real_par * epsilon_percent
            fitted_par = analysis.result().best_fit_parameters()

            self.assertTrue(fitted_par > real_par - epsilon and
                            fitted_par < real_par + epsilon)
