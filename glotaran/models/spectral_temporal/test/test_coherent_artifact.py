import numpy as np

from glotaran.model import ParameterGroup
from glotaran.models.spectral_temporal import KineticModel
from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix


def test_coherent_artifact():
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
            'irf1': {
                'type': 'gaussian',
                'center': '2',
                'width': '3',
                'coherent_artifact': True,
                'coherent_artifact_order': 3,
            },
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['mc1'],
                'irf': 'irf1',
            },
        },
    })

    parameter = ParameterGroup.from_list([
        101e-4,
        [10, {'vary': False, 'non-negative': False}],
        [20, {'vary': False, 'non-negative': False}],
    ])

    time = np.asarray(np.arange(0, 50, 1.5))
    dataset = model.dataset['dataset1'].fill(model, parameter)
    compartments, matrix = calculate_kinetic_matrix(dataset, 0, time)

    assert len(compartments) == 4
    for i in range(1, 4):
        assert compartments[i] == f'irf1_coherent_artifact_{i}'

    assert matrix.shape == (4, time.size)
