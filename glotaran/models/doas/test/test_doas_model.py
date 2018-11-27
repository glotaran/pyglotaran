import pytest
import numpy as np

from glotaran import ParameterGroup
from glotaran.models.doas import DOASModel
from glotaran.models.doas.doas_matrix import calculate_doas_matrix


def test_one_oscillation():
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

    print(sim_model.errors())
    assert sim_model.valid()

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

    print(model.errors())
    assert model.valid()

    wanted_parameter = ParameterGroup.from_dict({
        'osc': [
            ['freq', 500],
            ['rate', 0.1],
        ],
        'shape': {'amps': [7], 'locs': [5], 'width': [4]},
    })

    print(sim_model.errors_parameter(wanted_parameter))
    assert sim_model.valid_parameter(wanted_parameter)

    parameter = ParameterGroup.from_dict({
        'osc': [
            ['freq', 300],
            ['rate', 0.3],
        ],
    })

    print(model.errors_parameter(parameter))
    assert model.valid_parameter(parameter)

    dataset = sim_model.dataset['dataset1'].fill(sim_model, wanted_parameter)
    time = np.arange(0, 300)
    spectral = np.arange(0, 10)
    axis = {'time': time, 'spectral': spectral}

    clp, matrix = calculate_doas_matrix(dataset, 0, time)

    print(matrix.shape)
    assert matrix.shape == (2, 300)

    print(clp)
    assert clp == ['osc1_sin', 'osc1_cos']

    dataset = sim_model.simulate('dataset1', wanted_parameter, axis)
    print(dataset.data())

    assert dataset.data().shape == \
        (axis['spectral'].size, axis['time'].size)

    data = {'dataset1': dataset}

    result = model.fit(parameter, data)
    print(result.best_fit_parameter)

    for label, param in result.best_fit_parameter.all_with_label():
        assert np.allclose(param.value, parameter.get(label).value,
                           rtol=1e-1)

    resultdata = result.get_dataset("dataset1")
    assert np.array_equal(dataset.get_axis('time'), resultdata.get_axis('time'))
    assert np.array_equal(dataset.get_axis('spectral'), resultdata.get_axis('spectral'))
    assert dataset.data().shape == resultdata.data().shape
    assert np.allclose(dataset.data(), resultdata.data())
    assert False

def test_one_oscillation_two_compartment():
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
            },
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['m1']
            }
        }
    })

    print(model.errors())
    assert model.valid()
    parameter = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False}],
            ['0', 0, {'vary': False}],
        ],
        'kinetic': [
            ["1", 300e-3, {"min": 0}],
            ["2", 500e-4, {"min": 0}],
        ],
        'osc': [
            ['freq', 0.5],
            ['rate', 200e-2],
        ]
    })

    print(model.errors_parameter(parameter))
    assert model.valid_parameter(parameter)

    dataset = model.dataset['dataset1'].fill(model, parameter)
    axis = np.arange(0, 300.0)

    clp, matrix = calculate_doas_matrix(dataset, 0, axis)

    print(matrix.shape)
    assert matrix.shape == (4, 300)

    print(clp)
    assert clp == ['s1', 's2', 'osc1_sin', 'osc1_cos']
