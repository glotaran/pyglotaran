import pytest
import math
import numpy as np

from glotaran.model import ParameterGroup
from glotaran.models.spectral_temporal import KineticModel


class OneComponentOneChannel:
    model = KineticModel.from_dict({
        'compartment': ['s1'],
        'initial_concentration': {
            'j1': [['2']]
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {("s1", "s1"): '1', }}
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticModel.from_dict({
        'compartment': ['s1'],
        'initial_concentration': {
            'j1': [['2']]
        },
        'shape': {'sh1': ['one']},
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {("s1", "s1"): '1', }}
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['mc1'],
                'shapes': [['s1', 'sh1']]
            },
        },
    })

    initial = [101e-4, [1, {'vary': False}]]
    wanted = [101e-3, [1, {'vary': False}]]

    time = np.asarray(np.arange(0, 50, 1.5))
    spectral = np.asarray([0])
    axis = {"time": time, "spectral": spectral}

@pytest.mark.parametrize("suite", [
    OneComponentOneChannel,
])
def test_kinetic_model(suite):
    model = suite.model
    sim_model = suite.sim_model

    print(model.errors())
    assert model.valid()

    wanted = ParameterGroup.from_list(suite.wanted)
    print(sim_model.errors_parameter(wanted))
    print(wanted)
    assert sim_model.valid_parameter(wanted)

    initial = ParameterGroup.from_list(suite.initial)
    print(model.errors_parameter(initial))
    assert model.valid_parameter(initial)

    data = sim_model.simulate('dataset1', wanted, suite.axis)

    assert model.get_data('dataset1').get().shape == \
        (suite.axis['spectral'].size, suite.axis['time'].size)

    model.set_data('dataset1', data)

    result = fit(model, initial)
    print(result.best_fit_parameter)

    for param in result.best_fit_parameter.all():
        assert np.allclose(param.value, wanted.get(param.label).value,
                           rtol=1e-1)
#
#      def test_one_component_one_channel_gaussian_irf(self):
#          fitspec = '''
#  type: kinetic
#
#  compartments: [s1]
#
#  megacomplexes:
#      - label: mc1
#        k_matrices: [k1]
#
#  k_matrices:
#    - label: "k1"
#      matrix: {
#        '("s1","s1")': 1,
#  }
#
#  irf:
#    - label: irf1
#      type: gaussian
#      center: 2
#      width: 3
#
#  datasets:
#    - label: dataset1
#      type: spectral
#      megacomplexes: [mc1]
#      path: ''
#      irf: irf1
#
#  '''
#
#          initial_parameter = ParameterGroup.from_list([101e-4, 0, 5])
#          times = np.asarray(np.arange(0, 10, 1.5))
#          x = np.asarray([0])
#
#          wanted_params = ParameterGroup.from_list([101e-3, 0.3, 10])
#
#          model = parse_yml(fitspec)
#
#          axies = {"time": times, "spectral": x}
#
#          model.simulate(wanted_params, 'dataset1', axies)
#
#          result = model.fit(initial_parameter)
#          got_params = result.best_fit_parameter
#
#          for i in range(len(wanted_params)):
#              self.assertEpsilon(wanted_params["{}".format(i+1)],
#                                 got_params.get(f"{i+1}").value
#                                 )
#
#      def test_one_component_one_channel_gaussian_irf_convolve(self):
#          fitspec = '''
#  type: kinetic
#
#  compartments: [s1]
#
#  megacomplexes:
#      - label: mc1
#        k_matrices: [k1]
#
#  k_matrices:
#    - label: "k1"
#      matrix: {
#        '("s1","s1")': 1,
#  }
#
#  irf:
#    - label: irf1
#      type: measured
#
#  datasets:
#    - label: dataset1
#      type: spectral
#      megacomplexes: [mc1]
#      path: ''
#      irf: irf1
#
#  '''
#
#          initial_parameter = ParameterGroup.from_list([101e-4])
#          times = np.asarray(np.arange(0, 10, 1.5))
#          x = np.asarray([0])
#
#          center = 0
#          width = 5
#          irf = (1/np.sqrt(2 * np.pi)) * np.exp(-(times-center) * (times-center)
#                                                / (2 * width * width))
#
#          wanted_params = ParameterGroup.from_list([101e-3])
#
#          model = parse_yml(fitspec)
#          model.irfs["irf1"].data = irf
#
#          axies = {"time": times, "spectral": x}
#
#          model.simulate(wanted_params, 'dataset1', axies)
#
#          result = model.fit(initial_parameter)
#          got_params = result.best_fit_parameter
#
#          for i in range(len(wanted_params)):
#              self.assertEpsilon(wanted_params["{}".format(i+1)],
#                                 got_params.get(f"{i+1}").value
#                                 )
#
#      def test_three_component_sequential(self):
#          fitspec = '''
#  type: kinetic
#
#  compartments: [s1, s2, s3]
#
#  megacomplexes:
#      - label: mc1
#        k_matrices: [k1]
#
#  k_matrices:
#    - label: "k1"
#      matrix: {
#        '("s2","s1")': kinetic.1,
#        '("s3","s2")': kinetic.2,
#        '("s3","s3")': kinetic.3,
#  }
#
#  shapes:
#    - label: "shape1"
#      type: "gaussian"
#      amplitude: shape.amps.1
#      location: shape.locs.1
#      width: shape.width.1
#    - label: "shape2"
#      type: "gaussian"
#      amplitude: shape.amps.2
#      location: shape.locs.2
#      width: shape.width.2
#    - ["shape3", "gaussian", shape.amps.3, shape.locs.3, shape.width.3]
#
#  initial_concentration:
#    - label: jvec
#      parameter: [j.1, j.0, j.0]
#
#  irf:
#    - label: irf1
#      type: gaussian
#      center: irf.center
#      width: irf.width
#
#
#  datasets:
#    - label: dataset1
#      type: spectral
#      megacomplexes: [mc1]
#      irf: irf1
#      shapes:
#        - compartment: s1
#          shape: shape1
#        - [s2, shape2]
#        - [s3, shape3]
#
#  '''
#
#          amps = [3, 1, 5, False]
#          locations = [620, 670, 720, False]
#          delta = [10, 30, 50, False]
#          irf_center = 0
#          irf_width = 1
#
#          initial_parameter = [{'shape': [False,
#                                          {'amps': amps},
#                                          {'locs': locations},
#                                          {'width': delta}]},
#                               {'irf': [['center', irf_center],
#                                        ['width', irf_width]]},
#                               {'j': [['1', 1, {'vary': False}],
#                                      ['0', 0, {'vary': False}]]},
#                               ]
#
#          times = np.asarray(np.arange(-10, 100, 1.5))
#          x = np.arange(600, 750, 1)
#          axies = {"time": times, "spectral": x}
#
#          wanted_params = initial_parameter.copy()
#          wanted_params.append({"kinetic": [
#              ["1", 501e-4],
#              ["2", 202e-3],
#              ["3", 105e-2]]})
#          wanted_params = ParameterGroup.from_list(wanted_params)
#
#          model = parse_yml(fitspec)
#
#          model.simulate(wanted_params, 'dataset1', axies)
#
#          initial_parameter.append({'kinetic': [
#              ["1", 101e-4, {"min": 0}],
#              ["2", 202e-3, {"min": 0}],
#              ["3", 101e-1, {"min": 0}],
#          ]})
#          initial_parameter = ParameterGroup.from_list(initial_parameter)
#          result = model.fit(initial_parameter)
#          print(result.best_fit_parameter)
#
#          for param in wanted_params['kinetic'].all():
#              assert any([self.withinEpsilon(param.value, got.value)
#                          for got in result.best_fit_parameter.all()])
#
#      def test_three_component_multi_channel(self):
#          fitspec = '''
#  type: kinetic
#
#  compartments: [s1, s2, s3]
#
#  megacomplexes:
#      - label: mc1
#        k_matrices: [k1]
#
#  k_matrices:
#    - label: "k1"
#      matrix: {{
#        '("s1","s1")': 1,
#        '("s2","s2")': 2,
#        '("s3","s3")': 3,
#  }}
#
#  shapes:
#    - label: "shape1"
#      type: "gaussian"
#      amplitude: shape.amps.1
#      location: shape.locs.1
#      width: shape.width.1
#    - label: "shape2"
#      type: "gaussian"
#      amplitude: shape.amps.2
#      location: shape.locs.2
#      width: shape.width.2
#    - ["shape3", "gaussian", shape.amps.3, shape.locs.3, shape.width.3]
#
#  initial_concentrations: []
#
#  irf: []
#
#  datasets:
#    - label: dataset1
#      type: spectral
#      megacomplexes: [mc1]
#      path: ''
#      shapes:
#        - compartment: s1
#          shape: shape1
#        - [s2, shape2]
#        - [s3, shape3]
#
#  '''
#
#          amps = [7, 3, 30, False]
#          locations = [14700, 13515, 14180, False]
#          delta = [400, 100, 300, False]
#
#          initial_parameter = [{'shape': [False,
#                                          {'amps': amps},
#                                          {'locs': locations},
#                                          {'width': delta}]},
#                               ]
#
#          times = np.asarray(np.arange(-100, 1500, 1.5))
#          x = np.arange(12820, 15120, 4.6)
#          axies = {"time": times, "spectral": x}
#
#          wanted_params = initial_parameter.copy()
#          wanted_params.extend([101e-3, 202e-4, 305e-5])
#          wanted_params = ParameterGroup.from_list(wanted_params)
#
#          model = parse_yml(fitspec.format(initial_parameter))
#
#          model.simulate(wanted_params, 'dataset1', axies)
#
#          initial_parameter.extend([300e-3, 500e-4, 700e-5])
#          initial_parameter = ParameterGroup.from_list(initial_parameter)
#          result = model.fit(initial_parameter)
#
#          for param in wanted_params.all_group():
#              assert any([self.withinEpsilon(param.value, got.value)
#                          for got in result.best_fit_parameter.all()])
