import numpy as np
import xarray as xr

from glotaran.parameter import ParameterGroup
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.builtin.models.kinetic_spectrum.spectral_relations import \
    create_spectral_relation_matrix
from glotaran.builtin.models.kinetic_image.kinetic_image_matrix \
    import kinetic_image_matrix


def test_spectral_relation():
    model = KineticSpectrumModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3', 's4'],
                'parameters': ['i.1', 'i.2', 'i.3', 'i.4']
            },

        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s1", "s1"): 'kinetic.1',
                ("s2", "s2"): 'kinetic.1',
                ("s3", "s3"): 'kinetic.1',
                ("s4", "s4"): 'kinetic.1',
            }}
        },
        'spectral_relations': [
            {
                'compartment': 's1',
                'target': 's2',
                'parameter': 'rel.1',
                'interval': [(0, 2)],
            },
            {
                'compartment': 's1',
                'target': 's3',
                'parameter': 'rel.2',
                'interval': [(0, 2)],
            },
        ],
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['mc1'],
            },
        },
    })
    print(model)

    rel1, rel2 = 10, 20
    parameter = ParameterGroup.from_dict({
        'kinetic': [1e-4],
        'i': [1, 2, 3, 4],
        'rel': [rel1, rel2],
    })

    time = np.asarray(np.arange(0, 50, 1.5))
    dataset = model.dataset['dataset1'].fill(model, parameter)
    compartments, matrix = kinetic_image_matrix(dataset, time, 0)

    assert len(compartments) == 4
    assert matrix.shape == (time.size, 4)

    reduced_compartments, relation_matrix = \
        create_spectral_relation_matrix(model, parameter, compartments, matrix, 1)

    print(relation_matrix)
    assert len(reduced_compartments) == 2
    assert relation_matrix.shape == (4, 2)
    assert np.array_equal(
        relation_matrix,  [
            [1., 0.],
            [10., 0.],
            [20., 0.],
            [0., 1.],
        ]
    )

    reduced_compartments, reduced_matrix = \
        model.constrain_matrix_function(parameter, compartments, matrix, 1)

    assert reduced_matrix.shape == (time.size, 2)

    print(reduced_matrix[0, 0], matrix[0, 0], matrix[0, 1], matrix[0, 2])
    assert np.allclose(
        reduced_matrix[:, 0],  matrix[:, 0] + rel1 * matrix[:, 1] + rel2 * matrix[:, 2]
    )

    clp = xr.DataArray([[1., 10., 20., 1]],
                       coords=(('spectral', [1]), ('clp_label', ['s1', 's2', 's3', 's4'])))

    data = model.simulate('dataset1', parameter, clp=clp,
                          axes={'time': time, 'spectral': np.array([1])})

    result = model.optimize(parameter, {'dataset1': data}, max_nfev=1)

    result_data = result.data['dataset1']
    print(result_data.species_associated_spectra)
    assert result_data.species_associated_spectra.shape == (1, 4)
    assert result_data.species_associated_spectra[0, 1] == \
        rel1 * result_data.species_associated_spectra[0, 0]
    assert result_data.species_associated_spectra[0, 2] == \
        rel2 * result_data.species_associated_spectra[0, 0]


if __name__ == "__main__":
    test_spectral_relation()
