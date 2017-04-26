from unittest import TestCase
from copy import copy
import numpy as np

from lmfit import Parameters

from glotaran.specification_parser import parse_yml


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

io:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''

'''

        initial_parameter = [101e-4]
        times = np.asarray(np.arange(0, 1500, 1.5))
        x = np.asarray([0])

        wanted_params = Parameters()
        wanted_params.add("p_1", 101e-3)

        model = parse_yml(fitspec.format(initial_parameter))

        axies = {"time": times, "spectral": x}

        model.eval('dataset1', axies, parameter=wanted_params)

        result = model.fit()

        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params["p_{}".format(i+1)].value,
                               result.best_fit_parameter["p_{}".format(i+1)]
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

io:
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
        wanted_params.add("p_1", 101e-3)
        wanted_params.add("p_2", 0.3)
        wanted_params.add("p_3", 10)

        model = parse_yml(fitspec.format(initial_parameter))

        axies = {"time": times, "spectral": x}

        model.eval('dataset1', axies, parameter=wanted_params)

        result = model.fit()

        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params["p_{}".format(i+1)].value,
                               result.best_fit_parameter["p_{}".format(i+1)]
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

shapes:
  - label: "shape1"
    type: "gaussian"
    amplitude: shape.amps.1
    location: shape.locs.1
    width: shape.width.1
  - label: "shape2"
    type: "gaussian"
    amplitude: shape.amps.2
    location: shape.locs.2
    width: shape.width.2
  - ["shape3", "gaussian", shape.amps.3, shape.locs.3, shape.width.3]

initial_concentrations: []

irf: []

io:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''
    shapes:
      - compartment: s1
        shape: shape1
      - [s2, shape2]
      - [s3, shape3]

'''

        wanted_params = [101e-3, 202e-4, 305e-5]
        simparams = copy(wanted_params)
        times = np.asarray(np.arange(0, 1500, 1.5))
        x = np.arange(12820, 15120, 4.6)
        amps = [7, 3, 30, False]
        locations = [14700, 13515, 14180, False]
        delta = [400, 100, 300, False]

        simparams.append({'shape': [{'amps': amps}, {'locs': locations},
                         {'width': delta}]})

        model = parse_yml(fitspec.format(simparams))

        axies = {"time": times, "spectral": x}

        print(model.parameter.as_parameters_dict().pretty_print())

        model.eval('dataset1', axies)

        print(np.isnan(model.datasets['dataset1'].data.data).any())
        print(np.isnan(model.c_matrix()).any())
        model.parameter.get("1").value = 300e-3
        model.parameter.get("2").value = 500e-4
        model.parameter.get("3").value = 700e-5

        print(model.parameter.as_parameters_dict().pretty_print())
        result = model.fit()
        result.best_fit_parameter.pretty_print()
        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params[i],
                               result.best_fit_parameter["p_{}".format(i+1)]
                               .value, 1e-6)
