import pytest
import numpy as np

from glotaran.model import ParameterGroup
from glotaran.models.spectral_temporal import KineticModel


class OneComponentOneChannel:
    model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['2']
            },

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
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['2']
            },
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
                'shapes': {'s1': 'sh1'}
            },
        },
    })

    initial = ParameterGroup.from_list([101e-4, [1, {'vary': False, 'non-negative': False}]])
    wanted = ParameterGroup.from_list([101e-3, [1, {'vary': False, 'non-negative': False}]])

    time = np.asarray(np.arange(0, 50, 1.5))
    spectral = np.asarray([0])
    axis = {"time": time, "spectral": spectral}


class OneComponentOneChannelGaussianIrf:
    model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['2']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {("s1", "s1"): '1', }}
        },
        'irf': {
            'irf1': {'type': 'gaussian', 'center': '2', 'width': '3'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['4']
            },
        },
        'shape': {'sh1': ['one']},
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {("s1", "s1"): '1', }}
        },
        'irf': {
            'irf1': {'type': 'gaussian', 'center': '2', 'width': '3'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shapes': {'s1': 'sh1'}
            },
        },
    })

    initial = ParameterGroup.from_list([101e-4, 0.1, 5,
                                        [1, {'vary': False, 'non-negative': False}]])
    wanted = ParameterGroup.from_list([101e-3, 0.3, 10,
                                       [1, {'vary': False, 'non-negative': False}]])

    time = np.asarray(np.arange(-10, 50, 1.5))
    spectral = np.asarray([0])
    axis = {"time": time, "spectral": spectral}


class OneComponentOneChannelMeasuredIrf:
    model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['2']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {("s1", "s1"): '1', }}
        },
        'irf': {
            'irf1': {'type': 'measured'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['2']
            },
        },
        'shape': {'sh1': ['one']},
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {("s1", "s1"): '1', }}
        },
        'irf': {
            'irf1': {'type': 'measured'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shapes': {'s1': 'sh1'}
            },
        },
    })

    initial = ParameterGroup.from_list([101e-4, [1, {'vary': False, 'non-negative': False}]])
    wanted = ParameterGroup.from_list([101e-3, [1, {'vary': False, 'non-negative': False}]])

    time = np.asarray(np.arange(-10, 50, 1.5))
    spectral = np.asarray([0])
    axis = {"time": time, "spectral": spectral}

    center = 0
    width = 5
    irf = (1/np.sqrt(2 * np.pi)) * np.exp(-(time-center) * (time-center)
                                          / (2 * width * width))
    model.irf["irf1"].irfdata = irf
    sim_model.irf["irf1"].irfdata = irf


class ThreeComponentParallel:
    model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3'],
                'parameters': ['j.1', 'j.1', 'j.1']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s2", "s1"): 'kinetic.1',
                ("s3", "s2"): 'kinetic.2',
                ("s3", "s3"): 'kinetic.3',
            }}
        },
        'irf': {
            'irf1': {'type': 'gaussian', 'center': 'irf.center', 'width': 'irf.width'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3'],
                'parameters': ['j.1', 'j.1', 'j.1']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s1", "s1"): 'kinetic.1',
                ("s2", "s2"): 'kinetic.2',
                ("s3", "s3"): 'kinetic.3',
            }}
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
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shapes': {'s1': 'sh1', 's2': 'sh2', 's3': 'sh3'}
            },
        },
    })

    initial = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 300e-3],
            ["2", 500e-4],
            ["3", 700e-5],
        ],
        'irf': [['center', 1.1], ['width', 5]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}]],
    })
    wanted = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 101e-3],
            ["2", 202e-4],
            ["3", 305e-5],
        ],
        'shape': {'amps': [7, 3, 30], 'locs': [620, 670, 720], 'width': [10, 30, 50]},
        'irf': [['center', 1.3], ['width', 7.8]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}]],
    })

    time = np.arange(-10, 100, 1.5)
    spectral = np.arange(600, 750, 1)
    axis = {"time": time, "spectral": spectral}


class ThreeComponentSequential:
    model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3'],
                'parameters': ['j.1', 'j.0', 'j.0']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s2", "s1"): 'kinetic.1',
                ("s3", "s2"): 'kinetic.2',
                ("s3", "s3"): 'kinetic.3',
            }}
        },
        'irf': {
            'irf1': {'type': 'gaussian', 'center': 'irf.center', 'width': 'irf.width'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3'],
                'parameters': ['j.1', 'j.0', 'j.0']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s2", "s1"): 'kinetic.1',
                ("s3", "s2"): 'kinetic.2',
                ("s3", "s3"): 'kinetic.3',
            }}
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
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shapes': {'s1': 'sh1', 's2': 'sh2', 's3': 'sh3'}
            },
        },
    })

    initial = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 201e-3],
            ["2", 302e-4],
            ["3", 101e-5],
        ],
        'irf': [['center', 1.2], ['width', 7.5]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}],
              ['0', 0, {'vary': False, 'non-negative': False}]],
    })
    wanted = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 501e-3],
            ["2", 202e-4],
            ["3", 105e-5],
        ],
        'shape': {'amps': [3, 1, 5], 'locs': [620, 670, 720], 'width': [10, 30, 50]},
        'irf': [['center', 1.3], ['width', 7.8]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}],
              ['0', 0, {'vary': False, 'non-negative': False}]],
    })

    time = np.asarray(np.arange(-10, 50, 0.5))
    spectral = np.arange(600, 750, 1)
    axis = {"time": time, "spectral": spectral}


class IrfDispersion:
    model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3', 's4'],
                'parameters': ['j.1', 'j.0', 'j.0', 'j.0']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s2", "s1"): 'kinetic.1',
                ("s3", "s2"): 'kinetic.2',
                ("s4", "s3"): 'kinetic.3',
                ("s4", "s4"): 'kinetic.4',
            }}
        },
        'irf': {
            'irf1': {
                'type': 'gaussian',
                'center': 'irf.center',
                'width': 'irf.width',
                'dispersion_center': 'irf.dispcenter',
                'center_dispersion': ['irf.centerdisp'],
            },
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3', 's4'],
                'parameters': ['j.1', 'j.0', 'j.0', 'j.0']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s2", "s1"): 'kinetic.1',
                ("s3", "s2"): 'kinetic.2',
                ("s4", "s3"): 'kinetic.3',
                ("s4", "s4"): 'kinetic.4',
            }}
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
            'sh4': {
                'type': "gaussian",
                'amplitude': "shape.amps.4",
                'location': "shape.locs.4",
                'width': "shape.width.4",
            },
        },
        'irf': {
            'irf1': {
                'type': 'gaussian',
                'center': 'irf.center',
                'width': 'irf.width',
                'dispersion_center': 'irf.dispcenter',
                'center_dispersion': ['irf.centerdisp'],
            },
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shapes': {'s1': 'sh1', 's2': 'sh2', 's3': 'sh3', 's4': 'sh4'}
            },
        },
    })

    initial = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False, 'non-negative': False}],
            ['0', 0, {'vary': False, 'non-negative': False}],
        ],
        'kinetic': [
            ["1", 1],
            ["2", 0.4],
            ["3", 0.05],
            ["4", 0.009],
        ],
        'irf': [['center', 0.5],
                ['width', 0.3],
                ['dispcenter', 400, {'vary': False}],
                ['centerdisp', 0.25]],
    })
    wanted = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False, 'non-negative': False}],
            ['0', 0, {'vary': False, 'non-negative': False}],
        ],
        'kinetic': [
            ["1", 0.8],
            ["2", 0.2],
            ["3", 0.08],
            ["4", 0.01],
        ],

        'shape': {'amps': [2, 4, 5, 8], 'locs': [320, 380, 420, 460], 'width': [30, 20, 10, 40]},
        'irf': [['center', 0.3], ['width', 0.1], ['dispcenter', 400], ['centerdisp', 0.2]],
    })

    time = np.arange(-1, 30, 0.01)
    spectral = np.arange(300, 500, 1)
    axis = {"time": time, "spectral": spectral}


@pytest.mark.parametrize("suite", [
    OneComponentOneChannel,
    OneComponentOneChannelGaussianIrf,
    OneComponentOneChannelMeasuredIrf,
    ThreeComponentParallel,
    ThreeComponentSequential,
    IrfDispersion,
])
@pytest.mark.parametrize("nnls", [True, False])
def test_kinetic_model(suite, nnls):

    model = suite.model
    print(model.errors())
    assert model.valid()

    sim_model = suite.sim_model
    print(sim_model.errors())
    assert sim_model.valid()

    wanted = suite.wanted
    print(sim_model.errors_parameter(wanted))
    print(wanted)
    assert sim_model.valid_parameter(wanted)

    initial = suite.initial
    print(model.errors_parameter(initial))
    assert model.valid_parameter(initial)

    dataset = sim_model.simulate('dataset1', wanted, suite.axis)

    assert dataset.data().shape == \
        (suite.axis['time'].size, suite.axis['spectral'].size)

    data = {'dataset1': dataset}

    result = model.fit(initial, data, nnls=nnls, max_nfev=20)
    print(result.best_fit_parameter)

    for label, param in result.best_fit_parameter.all_with_label():
        assert np.allclose(param.value, wanted.get(label).value,
                           rtol=1e-1)

    resultdata = result.get_fitted_dataset("dataset1")
    assert np.array_equal(dataset.get_axis('time'), resultdata.get_axis('time'))
    assert np.array_equal(dataset.get_axis('spectral'), resultdata.get_axis('spectral'))
    assert dataset.data().shape == resultdata.data().shape
    assert np.allclose(dataset.data(), resultdata.data(), rtol=1e-2)
