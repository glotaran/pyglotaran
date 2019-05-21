import numpy as np
import pytest

from glotaran import ParameterGroup
from glotaran.models.doas import DOASModel
from glotaran.models.doas.doas_matrix import calculate_doas_matrix


@pytest.mark.skip
def test_doas_matrix_benchmark(benchmark):
    model = DOASModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2'],
                'parameters': ['j.1', 'j.0']
            },
        },
        'oscillation': {
            'osc1': {'frequency': 'osc.freq', 'rate': 'osc.rate'},
            'osc2': {'frequency': 'osc.freq', 'rate': 'osc.rate'}
        },
        'megacomplex': {
            'm1': {
                'oscillation': ['osc1', 'osc2'],
            }
        },
        'irf': {
            'irf1': {'type': 'gaussian', 'center': ['irf.center'], 'width': ['irf.width']},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['m1'],
                'irf': 'irf1',
            }
        }
    })
    parameter = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False}],
            ['0', 0, {'vary': False}],
        ],
        'osc': [
            ['freq', 16],
            ['rate', 0.3],
        ],
        'irf': [['center', 0.5], ['width', 0.3]],
    })
    time = np.arange(0, 6, 0.001)
    dataset = model.dataset['dataset1'].fill(model, parameter)

    benchmark(calculate_doas_matrix, dataset, time, 0)
