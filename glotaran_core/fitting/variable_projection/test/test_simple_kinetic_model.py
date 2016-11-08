from unittest import TestCase
from glotaran_core.fitting.variable_projection import (
    SeperableModel,
)
from lmfit import Parameters
import numpy as np


class TestSimpleKinetic(TestCase):

    def assertEpsilon(self, number, value, epsilon):
        self.assertTrue(abs(number - value) < epsilon)

    def test_one_compartment_decay(self):

        class OneComparmentDecay(SeperableModel):

            def c_matrix(self, parameter, *times, **kwargs):
                kinpar = np.asarray([parameter["p0"]])
                c = np.exp(np.outer(np.asarray(times), -kinpar))
                return c

            def e_matrix(self):
                # E Matrix => channels X compartments
                E = np.empty((1, 1), dtype=np.float64, order="F")

                E[0, 0] = 1

                return E

        model = OneComparmentDecay()
        times = np.asarray(np.arange(0, 1500, 1.5))

        params = [101e-4]

        real_params = Parameters()
        for i in range(len(params)):
            real_params.add("p{}".format(i), params[i])
        data = model.eval(real_params, times)

        initial_parameter = Parameters()
        initial_parameter.add("p0", 100e-5)

        result = model.fit(initial_parameter, *times, **{"data": data})
        for i in range(len(params)):
            self.assertEpsilon(params[i],
                               result.params["p{}".format(i)].value, 1e-6)

    def test_two_compartment_decay(self):

        class TwoComparmentDecay(SeperableModel):

            def c_matrix(self, parameter, *times, **kwargs):
                kinpar = np.asarray([parameter["p0"], parameter["p1"]])
                c = np.exp(np.outer(np.asarray(times), -kinpar))
                return c

            def e_matrix(self):
                # E Matrix => channels X compartments
                E = np.empty((1, 2), dtype=np.float64, order="F")

                E[0, 0] = 1
                E[0, 1] = 2
                return E

        model = TwoComparmentDecay()
        times = np.asarray(np.arange(0, 1500, 1.5))

        params = [101e-4, 202e-5]

        real_params = Parameters()
        for i in range(len(params)):
            real_params.add("p{}".format(i), params[i])

        data = model.eval(real_params.valuesdict(), *times)

        initial_parameter = Parameters()
        initial_parameter.add("p0", 100e-5)
        initial_parameter.add("p1", 200e-6)

        result = model.fit(initial_parameter, *times, **{"data": data})
        for i in range(len(params)):
            self.assertEpsilon(params[i],
                               result.params["p{}".format(i)].value,
                               1e-6)

    def test_multi_compartment_multi_channel_decay(self):

        class MultiChannelMultiCompartmentDecay(SeperableModel):

            def c_matrix(self, parameter, *times, **kwargs):
                kinpar = np.asarray([parameter["p{}".format(i)] for i in
                                     range(len((parameter)))])
                c = np.exp(np.outer(np.asarray(times), -kinpar))
                return c

            def e_matrix(self):
                wavenum = np.asarray(np.arange(12820, 15120, 4.6))
                location = np.asarray(
                    [14705, 13513, 14492, 14388, 14184, 13986])
                delta = np.asarray([400, 1000, 300, 200, 350, 330])
                amp = np.asarray([1, 0.2, 1, 1, 1, 1])

                E = np.empty((wavenum.size, location.size), dtype=np.float64,
                             order="F")

                for i in range(location.size):
                    E[:, i] = amp[i] * np.exp(
                        -np.log(2) * np.square(
                            2 * (wavenum - location[i])/delta[i]
                        )
                    )

                return E

        model = MultiChannelMultiCompartmentDecay()
        times = np.asarray(np.arange(0, 1500, 1.5))

        rparams = [.006667, .006667, 0.00333, 0.00035, 0.0303, 0.000909]

        real_params = Parameters()
        for i in range(len(rparams)):
            real_params.add("p{}".format(i), rparams[i])

        data = model.eval(real_params.valuesdict(), *times)

        params = [.005, 0.003, 0.00022, 0.0300, 0.000888]
        initial_parameter = Parameters()
        for i in range(len(params)):
            initial_parameter.add("p{}".format(i), params[i])

        result = model.fit(initial_parameter, *times, **{"data": data})

        wanted_params = [.006667, 0.00333, 0.00035, 0.0303, 0.000909]
        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params[i],
                               result.params["p{}".format(i)].value,
                               1e-6)
