import numpy as np
import xarray as xr

from glotaran.parameter import ParameterGroup
from glotaran.builtin.models.kinetic_image import KineticImageModel


def test_measured_irf():
    model = KineticImageModel.from_dict({
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
            'irf1': {'type': 'measured', 'method': 'conv1'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })

    initial = ParameterGroup.from_list([101e-4, [1, {'vary': False, 'non-negative': False}]])
    wanted = ParameterGroup.from_list([101e-3, [1, {'vary': False, 'non-negative': False}]])

    time = np.asarray(np.arange(-10, 50, 1.5))
    axis = {"time": time, "pixel": np.asarray([0])}

    center = 0
    width = 5
    irf = (1/np.sqrt(2 * np.pi)) * np.exp(-(time-center) * (time-center)
                                          / (2 * width * width))
    irf = xr.DataArray(irf, coords=[('time', time)])
    clp = xr.DataArray([[1]], coords=[('pixel', [0]), ('clp_label', ['s1'])])

    print(model.validate())
    assert model.valid()

    print(model.validate(wanted))
    print(wanted)
    assert model.valid(wanted)

    print(model.validate(initial))
    assert model.valid(initial)

    print(model.markdown(initial))

    extra = {'irf1': irf}

    dataset = model.simulate('dataset1', wanted, axis, clp, extra=extra)

    assert dataset.data.shape == \
        (axis['time'].size, axis['pixel'].size)

    data = {'dataset1': dataset}

    result = model.optimize(initial, data, max_nfev=20, extra=extra)
    print(result.optimized_parameter)

    for label, param in result.optimized_parameter.all():
        assert np.allclose(param.value, wanted.get(label).value,
                           rtol=1e-2)

    resultdata = result.data["dataset1"]
    assert np.array_equal(dataset['time'], resultdata['time'])
    assert np.array_equal(dataset['pixel'], resultdata['pixel'])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, rtol=1e-2)
