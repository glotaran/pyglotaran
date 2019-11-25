import copy

import numpy as np
import xarray as xr

from glotaran.parameter import ParameterGroup
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel


def test_spectral_penalties():

    model_without_penalty = KineticSpectrumModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3'],
                'parameters': ['i.1', 'i.2', 'i.3']
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
        'spectral_relations': [
            {
                'compartment': 's1',
                'target': 's2',
                'parameter': 'rel.1',
                'interval': [(0, 5)],  # try setting to [(0, 2)], [(0, 5)] and [(0, 10)]
            },
        ],
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['mc1'],
            },
        },
    })

    weight = 0.1
    model_with_penalty = KineticSpectrumModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3'],
                'parameters': ['i.1', 'i.2', 'i.3']
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
        'equal_area_penalties': [
            {
                'compartment': 's2',
                'target': 's3',
                'parameter': 'pen.1',
                # try setting to something other then spectral_relations['interval']
                'interval': [(0, 5)],
                'weight': weight
            },
        ],
        'spectral_relations': [
            {
                'compartment': 's1',
                'target': 's2',
                'parameter': 'rel.1',
                # try setting to [(0, 2)], [(0, 5)] and [(0, 10)]
                'interval': [(0, 5)],
            },
        ],
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['mc1'],
            },
        },
    })
    print(model_with_penalty)

    rel1 = 2
    pen = 0.5
    parameter = ParameterGroup.from_dict({
        'kinetic': [0.5, 0.01, 0.001],
        'i': [1, 1, 1],
        'rel': [rel1],
        'pen': [pen],
    })
    parameter2 = copy.deepcopy(parameter)
    del(parameter2['pen'])

    time_p1 = np.linspace(-1, 2, 50, endpoint=False)
    time_p2 = np.linspace(2, 10, 30, endpoint=False)
    time_p3 = np.geomspace(10, 50, num=20)
    time = np.concatenate([time_p1, time_p2, time_p3])
    spectral = np.linspace(1, 10, 10, endpoint=True)
    amps = np.transpose(np.tile([[2.], [1.], [3.]], [1, 10]))
    clp = xr.DataArray(amps,
                       coords=(('spectral', spectral), ('clp_label', ['s1', 's2', 's3'])))

    data = model_without_penalty.simulate('dataset1', parameter, clp=clp,
                                          axes={'time': time, 'spectral': spectral})

    result_without_penalty = \
        model_without_penalty.optimize(parameter2, {'dataset1': data}, max_nfev=1)

    result_with_penalty = \
        model_with_penalty.optimize(parameter, {'dataset1': data}, max_nfev=1)

    result_data = result_with_penalty.data['dataset1']
    wanted_penalty = result_data.species_associated_spectra.sel(species='s2') - \
        result_data.species_associated_spectra.sel(species='s3') * pen
    wanted_penalty *= weight
    wanted_penalty **= 2
    wanted_penalty = np.sum(wanted_penalty.values)

    additional_penalty = result_with_penalty.chisqr - result_without_penalty.chisqr
    assert np.isclose(additional_penalty, wanted_penalty)


if __name__ == "__main__":
    test_spectral_penalties()
