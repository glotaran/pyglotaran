import numpy as np
import pytest

from glotaran.parameter import ParameterGroup
from glotaran.analysis.result import Result
from glotaran.models.spectral_temporal import KineticModel
from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix


from .test_kinetic_model import ThreeComponentSequential


def test_kinetic_matrix_benchmark(benchmark):
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
            'irf1': {'type': 'gaussian', 'center': ['irf.center'], 'width': ['irf.width']},
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
            ["1", 101e-4],
            ["2", 302e-3],
            ["3", 201e-2],
        ],
        'irf': [['center', 0], ['width', 5]],
        'j': [['1', 1, {'vary': False}], ['0', 0, {'vary': False}]],
    })
    dataset = model.dataset['dataset1'].fill(model, parameter)
    time = np.asarray(np.arange(-10, 100, 0.02))

    benchmark(calculate_kinetic_matrix, dataset, 0, time)


@pytest.mark.parametrize("nnls", [True, False])
def test_kinetic_residual_benchmark(benchmark, nnls):

    suite = ThreeComponentSequential
    model = suite.model

    sim_model = suite.sim_model

    wanted = suite.wanted

    initial = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 501e-2],
            ["2", 202e-3],
            ["3", 105e-4],
        ],
        'irf': [['center', 0.3], ['width', 7.8]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}],
              ['0', 0, {'vary': False, 'non-negative': False}]],
    })

    dataset = sim_model.simulate('dataset1', wanted, suite.axis)

    data = {'dataset1': dataset}

    benchmark(Result.from_parameter, model, data, initial, nnls=nnls, atol=0)
