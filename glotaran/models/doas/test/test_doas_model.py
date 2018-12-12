import pytest
import numpy as np

from glotaran import ParameterGroup
from glotaran.models.doas import DOASModel
from glotaran.models.doas.doas_matrix import calculate_doas_matrix


class OneOscillation():
    sim_model = DOASModel.from_dict({
        'oscillation': {
            'osc1': {'frequency': 'osc.freq', 'rate': 'osc.rate'}
        },
        'megacomplex': {
            'm1': {'oscillation': ['osc1']}
        },
        'shape': {
            'sh1': {
                'type': "gaussian",
                'amplitude': "shape.amps.1",
                'location': "shape.locs.1",
                'width': "shape.width.1",
            },
        },
        'dataset': {
            'dataset1': {
                'megacomplex': ['m1'],
                'shapes': {'osc1': 'sh1'}
            }
        }
    })

    model = DOASModel.from_dict({
        'oscillation': {
            'osc1': {'frequency': 'osc.freq', 'rate': 'osc.rate'}
        },
        'megacomplex': {
            'm1': {'oscillation': ['osc1']}
        },
        'dataset': {
            'dataset1': {
                'megacomplex': ['m1']
            }
        }
    })

    wanted_parameter = ParameterGroup.from_dict({
        'osc': [
            ['freq', 25.5],
            ['rate', 0.1],
        ],
        'shape': {'amps': [7], 'locs': [5], 'width': [4]},
    })

    parameter = ParameterGroup.from_dict({
        'osc': [
            ['freq', 160],
            ['rate', 3],
        ],
    })

    time = np.arange(0, 3, 0.01)
    spectral = np.arange(0, 10)
    axis = {'time': time, 'spectral': spectral}

    wanted_clp = ['osc1_sin', 'osc1_cos']
    wanted_shape = (2, 300)


class OneOscillationWithIrf():
    sim_model = DOASModel.from_dict({
        'oscillation': {
            'osc1': {'frequency': 'osc.freq', 'rate': 'osc.rate'}
        },
        'megacomplex': {
            'm1': {'oscillation': ['osc1']}
        },
        'shape': {
            'sh1': {
                'type': "gaussian",
                'amplitude': "shape.amps.1",
                'location': "shape.locs.1",
                'width': "shape.width.1",
            },
        },
        'irf': {
            'irf1': {'type': 'gaussian', 'center': 'irf.center', 'width': 'irf.width'},
        },
        'dataset': {
            'dataset1': {
                'megacomplex': ['m1'],
                'shapes': {'osc1': 'sh1'},
                'irf': 'irf1',
            }
        }
    })

    model = DOASModel.from_dict({
        'oscillation': {
            'osc1': {'frequency': 'osc.freq', 'rate': 'osc.rate'}
        },
        'megacomplex': {
            'm1': {'oscillation': ['osc1']}
        },
        'irf': {
            'irf1': {'type': 'gaussian', 'center': 'irf.center', 'width': 'irf.width'},
        },
        'dataset': {
            'dataset1': {
                'megacomplex': ['m1'],
                'irf': 'irf1',
            }
        }
    })

    wanted_parameter = ParameterGroup.from_dict({
        'osc': [
            ['freq', 25.5],
            ['rate', 0.1],
        ],
        'shape': {'amps': [7], 'locs': [5], 'width': [4]},
        'irf': [['center', 0.3], ['width', 0.1]],
    })

    parameter = ParameterGroup.from_dict({
        'osc': [
            ['freq', 16],
            ['rate', 0.3],
        ],
        'irf': [['center', 0.5], ['width', 0.2]],
    })

    time = np.arange(0, 3, 0.01)
    spectral = np.arange(0, 10)
    axis = {'time': time, 'spectral': spectral}

    wanted_clp = ['osc1_sin', 'osc1_cos']
    wanted_shape = (2, 300)


class OneOscillationWithSequentialModel():
    sim_model = DOASModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2'],
                'parameters': ['j.1', 'j.0']
            },
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s2", "s1"): 'kinetic.1',
                ("s2", "s2"): 'kinetic.2',
            }}
        },
        'oscillation': {
            'osc1': {'frequency': 'osc.freq', 'rate': 'osc.rate'}
        },
        'megacomplex': {
            'm1': {
                'k_matrix': ['k1'],
                'oscillation': ['osc1'],
            }
        },
        'shape': {
            'sh1': {
                'type': "gaussian",
                'amplitude': "shape.amps.1",
                'location': "shape.locs.1",
                'width': "shape.width.1",
            },
            'sh2': {
                'type': "gaussian",
                'amplitude': "shape.amps.2",
                'location': "shape.locs.2",
                'width': "shape.width.2",
            },
            'sh3': {
                'type': "gaussian",
                'amplitude': "shape.amps.3",
                'location': "shape.locs.3",
                'width': "shape.width.3",
            },
        },
        'irf': {
            'irf1': {'type': 'gaussian', 'center': 'irf.center', 'width': 'irf.width'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['m1'],
                'shapes': {
                    'osc1': 'sh1',
                    's1': 'sh2',
                    's2': 'sh3',
                },
                'irf': 'irf1',
            }
        }
    })

    model = DOASModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2'],
                'parameters': ['j.1', 'j.0']
            },
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s2", "s1"): 'kinetic.1',
                ("s2", "s2"): 'kinetic.2',
            }}
        },
        'oscillation': {
            'osc1': {'frequency': 'osc.freq', 'rate': 'osc.rate'}
        },
        'megacomplex': {
            'm1': {
                'k_matrix': ['k1'],
                'oscillation': ['osc1'],
            }
        },
        'irf': {
            'irf1': {'type': 'gaussian', 'center': 'irf.center', 'width': 'irf.width'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['m1'],
                'irf': 'irf1',
            }
        }
    })

    wanted_parameter = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False, 'non-negative': False}],
            ['0', 0, {'vary': False, 'non-negative': False}],
        ],
        'kinetic': [
            ["1", 0.2],
            ["2", 0.01],
        ],
        'osc': [
            ['freq', 25.5],
            ['rate', 0.1],
        ],
        'shape': {'amps': [0.07, 2, 4], 'locs': [5, 2, 8], 'width': [4, 2, 3]},
        'irf': [['center', 0.3], ['width', 0.5]],
    })

    parameter = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False, 'non-negative': False}],
            ['0', 0, {'vary': False, 'non-negative': False}],
        ],
        'kinetic': [
            ["1", 0.3],
            ["2", 0.05],
        ],
        'osc': [
            ['freq', 16],
            ['rate', 0.3],
        ],
        'irf': [['center', 0.5], ['width', 0.3]],
    })

    time = np.arange(-1, 5, 0.01)
    spectral = np.arange(0, 10)
    axis = {'time': time, 'spectral': spectral}

    wanted_clp = ['osc1_sin', 'osc1_cos', 's1', 's2']
    wanted_shape = (4, 600)


@pytest.mark.parametrize("suite", [
    OneOscillation,
    OneOscillationWithIrf,
    OneOscillationWithSequentialModel,
])
def test_doas_model(suite):

    print(suite.sim_model.errors())
    assert suite.sim_model.valid()

    print(suite.model.errors())
    assert suite.model.valid()

    print(suite.sim_model.errors_parameter(suite.wanted_parameter))
    assert suite.sim_model.valid_parameter(suite.wanted_parameter)

    print(suite.model.errors_parameter(suite.parameter))
    assert suite.model.valid_parameter(suite.parameter)

    dataset = suite.sim_model.dataset['dataset1'].fill(suite.sim_model, suite.wanted_parameter)

    clp, matrix = calculate_doas_matrix(dataset, 0, suite.time)

    print(matrix.shape)
    assert matrix.shape == suite.wanted_shape

    print(clp)
    assert clp == suite.wanted_clp

    dataset = suite.sim_model.simulate('dataset1', suite.wanted_parameter,
                                       suite.axis)
    print(dataset.data())

    assert dataset.data().shape == \
        (suite.axis['spectral'].size, suite.axis['time'].size)

    print(suite.parameter)
    data = {'dataset1': dataset}

    result = suite.model.fit(suite.parameter, data)
    print(result.best_fit_parameter)

    for label, param in result.best_fit_parameter.all_with_label():
        assert np.allclose(param.value, suite.wanted_parameter.get(label).value,
                           rtol=1e-1)

    resultdata = result.get_fitted_dataset("dataset1")
    assert np.array_equal(dataset.get_axis('time'), resultdata.get_axis('time'))
    assert np.array_equal(dataset.get_axis('spectral'), resultdata.get_axis('spectral'))
    assert dataset.data().shape == resultdata.data().shape
    assert np.allclose(dataset.data(), resultdata.data())
