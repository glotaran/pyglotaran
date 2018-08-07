from unittest import TestCase
# unused import
# from copy import copy
import math
import numpy as np

# unused import
# from lmfit import Parameters

from glotaran.specification_parser import parse_yml


class TestKineticModel(TestCase):

    def assertEpsilon(self, wanted_value, given_value):
        min_want = np.min(np.abs(wanted_value))
        epsilon = 10**(math.floor(math.log10(min_want)) - 3)
        msg = 'Want: {} Have: {} with epsilon {}'.format(wanted_value, given_value, epsilon)
        assert self.withinEpsilon(wanted_value, given_value), msg

    def withinEpsilon(self, wanted_value, given_value):
        min_want = np.min(np.abs(wanted_value))
        epsilon = 10**(math.floor(math.log10(min_want)) - 3)
        return np.any(np.abs(wanted_value - given_value) < epsilon)

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

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''

'''

        initial_parameter = [101e-4]
        times = np.asarray(np.arange(0, 50, 1.5))
        x = np.asarray([0])

        wanted_params = {"1": 101e-3}

        model = parse_yml(fitspec.format(initial_parameter))

        axies = {"time": times, "spectral": x}

        model.simulate('dataset1', axies, parameter=wanted_params)

        result = model.fit()
        got_params = result.best_fit_parameter

        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params["{}".format(i+1)],
                               got_params.get(f"{i+1}").value
                               )

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
        times = np.asarray(np.arange(0, 10, 1.5))
        x = np.asarray([0])

        wanted_params = {"1": 101e-3, "2": 0.3, "3": 10}

        model = parse_yml(fitspec.format(initial_parameter))

        axies = {"time": times, "spectral": x}

        model.simulate('dataset1', axies, parameter=wanted_params)

        result = model.fit()
        got_params = result.best_fit_parameter

        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params["{}".format(i+1)],
                               got_params.get(f"{i+1}").value
                               )
    def test_one_component_one_channel_gaussian_irf_convolve(self):
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

irf:
  - label: irf1
    type: measured

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''
    irf: irf1

'''

        initial_parameter = [101e-4]
        times = np.asarray(np.arange(0, 10, 1.5))
        x = np.asarray([0])

        center = 0
        width = 5
        irf = (1/np.sqrt(2 * np.pi)) * np.exp(-(times-center) * (times-center)
                                              / (2 * width * width))

        wanted_params = {"1": 101e-3}

        model = parse_yml(fitspec.format(initial_parameter))
        model.irfs["irf1"].data = irf

        axies = {"time": times, "spectral": x}

        model.simulate('dataset1', axies, parameter=wanted_params)

        result = model.fit()
        got_params = result.best_fit_parameter

        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params["{}".format(i+1)],
                               got_params.get(f"{i+1}").value
                               )

    def test_three_component_sequential(self):
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
      '("s2","s1")': kinetic.1,
      '("s3","s2")': kinetic.2,
      '("s3","s3")': kinetic.3,
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

initial_concentration:
  - label: jvec
    parameter: [j.1, j.0, j.0]

irf:
  - label: irf1
    type: gaussian
    center: irf.center
    width: irf.width


datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    irf: irf1
    shapes:
      - compartment: s1
        shape: shape1
      - [s2, shape2]
      - [s3, shape3]

'''

        initial_parameter = [{'kinetic': [
            ["1", 101e-4, {"min": 0}],
            ["2", 202e-3, {"min": 0}],
            ["3", 101e-1, {"min": 0}],
        ]}]
        amps = [3, 1, 5, False]
        locations = [620, 670, 720, False]
        delta = [10, 30, 50, False]
        irf_center = 0
        irf_width = 1

        initial_parameter.append({'shape': [False, {'amps': amps}, {'locs': locations},
                                 {'width': delta}]})

        initial_parameter.append({'irf': [['center', irf_center], ['width', irf_width]]})
        initial_parameter.append({'j': [['1', 1, {'vary': False}], ['0', 0,{'vary': False}]]})

        times = np.asarray(np.arange(-10, 100, 1.5))
        x = np.arange(600, 750, 1)
        axies = {"time": times, "spectral": x}

        wanted_params = {"kinetic.1": 501e-4, "kinetic.2": 202e-3, "kinetic.3": 105e-2}

        model = parse_yml(fitspec.format(initial_parameter))

        model.simulate('dataset1', axies, parameter=wanted_params)

        result = model.fit()
        print(result.best_fit_parameter)

        for i in wanted_params:
            param = wanted_params[i]
            assert any([self.withinEpsilon(param, got.value)
                        for got in result.best_fit_parameter.all()])

    def test_one_component_one_channel_gaussian_irf_convolve(self):
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

irf:
  - label: irf1
    type: measured

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''
    irf: irf1

'''

        initial_parameter = [101e-4]
        times = np.asarray(np.arange(0, 10, 1.5))
        x = np.asarray([0])

        center = 0
        width = 5
        irf = (1/np.sqrt(2 * np.pi)) * np.exp(-(times-center) * (times-center)
                                              / (2 * width * width))

        wanted_params = {"1": 101e-3}

        model = parse_yml(fitspec.format(initial_parameter))
        model.irfs["irf1"].data = irf

        axies = {"time": times, "spectral": x}

        model.simulate('dataset1', axies, parameter=wanted_params)

        result = model.fit()
        got_params = result.best_fit_parameter

        for i in range(len(wanted_params)):
            self.assertEpsilon(wanted_params["{}".format(i+1)],
                               got_params.get(f"{i+1}").value
                               )

    def test_three_component_sequential(self):
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
      '("s2","s1")': kinetic.1,
      '("s3","s2")': kinetic.2,
      '("s3","s3")': kinetic.3,
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

initial_concentration:
  - label: jvec
    parameter: [j.1, j.0, j.0]

irf:
  - label: irf1
    type: gaussian
    center: irf.center
    width: irf.width


datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    irf: irf1
    shapes:
      - compartment: s1
        shape: shape1
      - [s2, shape2]
      - [s3, shape3]

'''

        initial_parameter = [{'kinetic': [
            ["1", 101e-4, {"min": 0}],
            ["2", 202e-3, {"min": 0}],
            ["3", 101e-1, {"min": 0}],
        ]}]
        amps = [3, 1, 5, False]
        locations = [620, 670, 720, False]
        delta = [10, 30, 50, False]
        irf_center = 0
        irf_width = 1

        initial_parameter.append({'shape': [False, {'amps': amps}, {'locs': locations},
                                 {'width': delta}]})

        initial_parameter.append({'irf': [['center', irf_center], ['width', irf_width]]})
        initial_parameter.append({'j': [['1', 1, {'vary': False}], ['0', 0, {'vary': False}]]})

        times = np.asarray(np.arange(-10, 100, 1.5))
        x = np.arange(600, 750, 1)
        axies = {"time": times, "spectral": x}

        wanted_params = {"kinetic.1": 501e-4, "kinetic.2": 202e-3, "kinetic.3": 105e-2}

        model = parse_yml(fitspec.format(initial_parameter))

        model.simulate('dataset1', axies, parameter=wanted_params)

        result = model.fit()
        print(result.best_fit_parameter)

        for i in wanted_params:
            param = wanted_params[i]
            assert any([self.withinEpsilon(param, got.value)
                        for got in result.best_fit_parameter.all()])

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

datasets:
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

        initial_parameter = [300e-3, 500e-4, 700e-5]
        amps = [7, 3, 30, False]
        locations = [14700, 13515, 14180, False]
        delta = [400, 100, 300, False]

        initial_parameter.append({'shape': [False, {'amps': amps}, {'locs': locations},
                                 {'width': delta}]})

        times = np.asarray(np.arange(-100, 1500, 1.5))
        x = np.arange(12820, 15120, 4.6)
        axies = {"time": times, "spectral": x}

        wanted_params = {"1": 101e-3, "2": 202e-4, "3": 305e-5}

        model = parse_yml(fitspec.format(initial_parameter))

        model.simulate('dataset1', axies, parameter=wanted_params)

        result = model.fit()

        for i in wanted_params:
            param = wanted_params[i]
            assert any([self.withinEpsilon(param, got.value)
                        for got in result.best_fit_parameter.all()])
