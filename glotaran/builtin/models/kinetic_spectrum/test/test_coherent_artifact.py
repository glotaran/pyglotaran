import numpy as np
import xarray as xr

from glotaran.parameter import ParameterGroup
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_matrix \
    import kinetic_spectrum_matrix


def test_coherent_artifact():
    model_dict = {
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
                'type': 'gaussian-coherent-artifact',
                'center': '2',
                'width': '3',
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
    }
    model = KineticSpectrumModel.from_dict(model_dict.copy())

    parameter = ParameterGroup.from_list([
        101e-4,
        [10, {'vary': False, 'non-negative': False}],
        [20, {'vary': False, 'non-negative': False}],
        [30, {'vary': False, 'non-negative': False}],
    ])

    time = np.asarray(np.arange(0, 50, 1.5))

    irf = model.irf['irf1'].fill(model, parameter)
    irf_same_width = irf.calculate_coherent_artifact(time)

    model_dict['irf']['irf1']['coherent_artifact_width'] = '4'
    model = KineticSpectrumModel.from_dict(model_dict)

    irf = model.irf['irf1'].fill(model, parameter)
    irf_diff_width = irf.calculate_coherent_artifact(time)

    assert not np.array_equal(irf_same_width, irf_diff_width)

    dataset = model.dataset['dataset1'].fill(model, parameter)
    compartments, matrix = kinetic_spectrum_matrix(dataset, time, 0)

    assert len(compartments) == 4
    for i in range(1, 4):
        assert compartments[i] == f'coherent_artifact_{i}'

    assert matrix.shape == (time.size, 4)

    clp = xr.DataArray([[1, 1, 1, 1]],
                       coords=[('spectral', [0]),
                               ('clp_label', ['s1',
                                              'coherent_artifact_1',
                                              'coherent_artifact_2',
                                              'coherent_artifact_3',
                                              ])])
    axis = {"time": time, "spectral": clp.spectral}
    dataset = model.simulate('dataset1', parameter, axis, clp)

    data = {'dataset1': dataset}
    result = model.optimize(parameter, data, max_nfev=20)
    print(result.optimized_parameter)

    for label, param in result.optimized_parameter.all():
        assert np.allclose(param.value, parameter.get(label).value,
                           rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(dataset['time'], resultdata['time'])
    assert np.array_equal(dataset['spectral'], resultdata['spectral'])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, rtol=1e-2)

    assert 'coherent_artifact_concentration' in resultdata
    assert resultdata['coherent_artifact_concentration'].shape == (time.size, 3)

    assert 'coherent_artifact_associated_spectra' in resultdata
    assert resultdata['coherent_artifact_associated_spectra'].shape == (1, 3)
