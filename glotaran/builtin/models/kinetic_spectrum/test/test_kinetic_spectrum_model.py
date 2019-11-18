import pytest
import numpy as np

from glotaran.parameter import ParameterGroup
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel


class OneComponentOneChannel:
    model = KineticSpectrumModel.from_dict({
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
    sim_model = KineticSpectrumModel.from_dict({
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
                'shape': {'s1': 'sh1'}
            },
        },
    })

    initial = ParameterGroup.from_list([101e-4, [1, {'vary': False, 'non-negative': False}]])
    wanted = ParameterGroup.from_list([101e-3, [1, {'vary': False, 'non-negative': False}]])

    time = np.asarray(np.arange(0, 50, 1.5))
    spectral = np.asarray([0])
    axis = {"time": time, "spectral": spectral}


class OneComponentOneChannelGaussianIrf:
    model = KineticSpectrumModel.from_dict({
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
            'irf1': {'type': 'spectral-gaussian', 'center': '2', 'width': '3'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticSpectrumModel.from_dict({
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
            'irf1': {'type': 'spectral-gaussian', 'center': '2', 'width': '3'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shape': {'s1': 'sh1'}
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


class ThreeComponentParallel:
    model = KineticSpectrumModel.from_dict({
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
            'irf1': {
                'type': 'spectral-multi-gaussian',
                'center': ['irf.center'],
                'width': ['irf.width']},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticSpectrumModel.from_dict({
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
            'irf1': {
                'type': 'spectral-multi-gaussian',
                'center': ['irf.center'],
                'width': ['irf.width']
            },
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shape': {'s1': 'sh1', 's2': 'sh2', 's3': 'sh3'}
            },
        },
    })

    initial = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 300e-3],
            ["2", 500e-4],
            ["3", 700e-5],
        ],
        'irf': [['center', 1.3], ['width', 7.8]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}]],
    })
    wanted = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 301e-3],
            ["2", 502e-4],
            ["3", 705e-5],
        ],
        'shape': {'amps': [7, 3, 30], 'locs': [620, 670, 720], 'width': [10, 30, 50]},
        'irf': [['center', 1.3], ['width', 7.8]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}]],
    })

    time = np.arange(-10, 100, 1.5)
    spectral = np.arange(600, 750, 10)
    axis = {"time": time, "spectral": spectral}


class ThreeComponentSequential:
    model = KineticSpectrumModel.from_dict({
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
            'irf1': {
                'type': 'spectral-multi-gaussian',
                'center': ['irf.center'],
                'width': ['irf.width']
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
    sim_model = KineticSpectrumModel.from_dict({
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
            'irf1': {'type': 'spectral-multi-gaussian',
                     'center': ['irf.center'],
                     'width': ['irf.width']},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shape': {'s1': 'sh1', 's2': 'sh2', 's3': 'sh3'}
            },
        },
    })

    initial = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 501e-3],
            ["2", 202e-4],
            ["3", 105e-5],
            {'non-negative': True},
        ],
        'irf': [['center', 1.3], ['width', 7.8]],
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

    time = np.asarray(np.arange(-10, 50, 1.0))
    spectral = np.arange(600, 750, 5.0)
    axis = {"time": time, "spectral": spectral}


@pytest.mark.parametrize("suite", [
    OneComponentOneChannel,
    OneComponentOneChannelGaussianIrf,
    ThreeComponentParallel,
    ThreeComponentSequential,
])
@pytest.mark.parametrize("nnls", [True, False])
def test_kinetic_model(suite, nnls):

    model = suite.model
    print(model.validate())
    assert model.valid()

    sim_model = suite.sim_model
    print(sim_model.validate())
    assert sim_model.valid()

    wanted = suite.wanted
    print(sim_model.validate(wanted))
    print(wanted)
    assert sim_model.valid(wanted)

    initial = suite.initial
    print(model.validate(initial))
    assert model.valid(initial)

    print(model.markdown(wanted))

    dataset = sim_model.simulate('dataset1', wanted, suite.axis)

    assert dataset.data.shape == \
        (suite.axis['time'].size, suite.axis['spectral'].size)

    data = {'dataset1': dataset}

    result = model.optimize(initial, data, nnls=nnls, max_nfev=20)
    print(result.optimized_parameter)

    for label, param in result.optimized_parameter.all():
        assert np.allclose(param.value, wanted.get(label).value,
                           rtol=1e-1)

    resultdata = result.data["dataset1"]

    print(resultdata)

    assert np.array_equal(dataset['time'], resultdata['time'])
    assert np.array_equal(dataset['spectral'], resultdata['spectral'])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, rtol=1e-2)

    assert 'species_associated_spectra' in resultdata
    assert 'decay_associated_spectra' in resultdata
