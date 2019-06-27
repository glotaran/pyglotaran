import numpy as np

from glotaran.parameter import ParameterGroup
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_matrix \
    import kinetic_spectrum_matrix


def test_coherent_artifact():
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
            'irf1': {
                'type': 'spectral-gaussian',
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
    compartments, matrix = kinetic_spectrum_matrix(dataset, time, 0)

    assert len(compartments) == 4
    for i in range(1, 4):
        assert compartments[i] == f'irf1_coherent_artifact_{i}'

    assert matrix.shape == (time.size, 4)
