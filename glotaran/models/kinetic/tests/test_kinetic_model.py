from unittest import TestCase
import numpy as np

from lmfit import Parameters

from glotaran.specification_parser import parse_yml
from glotaran.model import IndependentAxies
from glotaran.models.kinetic import KineticSeparableModel


class TestKineticModel(TestCase):

    def assertEpsilon(self, number, value, epsilon):
        self.assertTrue(abs(number - value) < epsilon,
                        msg='Want: {} Have: {}'.format(number, value))

    def test_one_component_one_channel(self):
        fitspec = '''
type: kinetic

parameters: {}

compartments: [s1]

megacomplexes:
    - label: mc1
      k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {{
      '("s1","s1")': 1,
}}

initial_concentrations: []

irf: []

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''

'''

        initial_parameter = [101e-4]
        times = np.asarray(np.arange(0, 1500, 1.5))
        x = np.asarray([0])

        wanted_params = Parameters()
        wanted_params.add("p1", 101e-3)

        model = parse_yml(fitspec.format(initial_parameter))

        axies = IndependentAxies()
        axies.add(x)
        axies.add(times)

        data = model.eval(wanted_params, 'dataset1', axies)

        fitmodel = KineticSeparableModel(model)
        result = fitmodel.fit(fitmodel.get_initial_fitting_parameter(),
                              *times, **{"dataset1": data})

        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params["p{}".format(i+1)].value,
                               result.best_fit_parameter["p{}".format(i+1)]
                               .value, 1e-6)

    def test_one_component_one_channel_gaussian_irf(self):
        fitspec = '''
type: kinetic

parameters: {}

compartments: [s1]

megacomplexes:
    - label: mc1
      k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {{
      '("s1","s1")': 1,
}}

initial_concentrations: []

irf:
  - label: irf1
    type: gaussian
    center: 2
    width: 3

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''
    irf: irf1

'''

        initial_parameter = [101e-4, 0, 5]
        times = np.asarray(np.arange(-100, 1500, 1.5))
        x = np.asarray([0])

        wanted_params = Parameters()
        wanted_params.add("p1", 101e-3)
        wanted_params.add("p2", 0.3)
        wanted_params.add("p3", 10)

        model = parse_yml(fitspec.format(initial_parameter))

        axies = IndependentAxies()
        axies.add(x)
        axies.add(times)

        data = model.eval(wanted_params, 'dataset1', axies)

        fitmodel = KineticSeparableModel(model)
        result = fitmodel.fit(fitmodel.get_initial_fitting_parameter(),
                              *times, **{"dataset1": data})

        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params["p{}".format(i+1)].value,
                               result.best_fit_parameter["p{}".format(i+1)]
                               .value, 1e-6)

    def test_three_component_multi_channel(self):
        fitspec = '''
type: kinetic

parameters: {}

compartments: [s1, s2, s3]

megacomplexes:
    - label: mc1
      k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {{
      '("s1","s1")': 1,
      '("s2","s2")': 2,
      '("s3","s3")': 3,
}}

initial_concentrations: []

irf: []

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''

'''

        initial_parameter = [301e-3, 502e-4, 205e-5]
        times = np.asarray(np.arange(0, 1500, 1.5))
        x = np.arange(12820, 15120, 4.6)
        amps = [7, 3, 30]
        locations = [14700, 13515, 14180]
        delta = [400, 100, 300]

        wanted_params = Parameters()
        wanted_params.add("p1", 101e-3)
        wanted_params.add("p2", 202e-4)
        wanted_params.add("p3", 505e-5)

        model = parse_yml(fitspec.format(initial_parameter))

        axies = IndependentAxies()
        axies.add(x)
        axies.add(times)

        data = model.eval(wanted_params, 'dataset1', axies,
                          **{'amplitudes': amps,
                             'locations': locations,
                             'delta': delta})

        fitmodel = KineticSeparableModel(model)

        result = fitmodel.fit(fitmodel.get_initial_fitting_parameter(),
                              *times, **{"dataset1": data})
        result.best_fit_parameter.pretty_print()
        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params["p{}".format(i+1)].value,
                               result.best_fit_parameter["p{}".format(i+1)]
                               .value, 1e-6)
