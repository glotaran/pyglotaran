import numpy as np

from glotaran import ParameterGroup
from glotaran.models.spectral_temporal import KineticModel
from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix


def test_kinetic_benchmark(benchmark):
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
    parameter = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 101e-4, {"min": 0}],
            ["2", 302e-3, {"min": 0}],
            ["3", 201e-2, {"min": 0}],
        ],
        'irf': [['center', 0], ['width', 5]],
        'j': [['1', 1, {'vary': False}], ['0', 0, {'vary': False}]],
    })
    dataset = model.dataset['dataset1'].fill(model, parameter)
    time = np.asarray(np.arange(-10, 100, 0.02))

    benchmark(calculate_kinetic_matrix, dataset, 0, time)
