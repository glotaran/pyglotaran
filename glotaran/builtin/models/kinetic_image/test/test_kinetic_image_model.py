import pytest
import numpy as np
import xarray as xr

from glotaran.parameter import ParameterGroup
from glotaran.builtin.models.kinetic_image import KineticImageModel


def _create_gaussian_clp(labels, amplitudes, centers, widths, axis):
    return xr.DataArray(
        [amplitudes[i] * np.exp(-np.log(2) * np.square(2 * (axis - centers[i]) / widths[i]))
         for i, _ in enumerate(labels)], coords=[('clp_label', labels), ('pixel', axis)]
    ).T


class OneComponentOneChannel:
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
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'megacomplex': ['mc1'],
            },
        },
    })

    initial = ParameterGroup.from_list([101e-4, [1, {'vary': False, 'non-negative': False}]])
    wanted = ParameterGroup.from_list([101e-3, [1, {'vary': False, 'non-negative': False}]])

    time = np.asarray(np.arange(0, 50, 1.5))
    axis = {"time": time, "pixel": np.asarray([0])}

    clp = xr.DataArray([[1]], coords=[('pixel', [0]), ('clp_label', ['s1'])])


class OneComponentOneChannelGaussianIrf:
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
            'irf1': {'type': 'gaussian', 'center': '2', 'width': '3'},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })

    initial = ParameterGroup.from_list([101e-4, 0.1, 5,
                                        [1, {'vary': False, 'non-negative': False}]])
    wanted = ParameterGroup.from_list([101e-3, 0.3, 10,
                                       [1, {'vary': False, 'non-negative': False}]])

    time = np.asarray(np.arange(-10, 50, 1.5))
    axis = {"time": time, "pixel": np.asarray([0])}
    clp = xr.DataArray([[1]], coords=[('pixel', [0]), ('clp_label', ['s1'])])


class OneComponentOneChannelMeasuredIrf:
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
            'irf1': {'type': 'measured'},
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
    model.irf["irf1"].irfdata = irf

    clp = xr.DataArray([[1]], coords=[('pixel', [0]), ('clp_label', ['s1'])])


class ThreeComponentParallel:
    model = KineticImageModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1', 's2', 's3'],
                'parameters': ['j.1', 'j.1', 'j.1']
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
            'irf1': {'type': 'multi-gaussian', 'center': ['irf.center'], 'width': ['irf.width']},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })

    initial = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 300e-3],
            ["2", 500e-4],
            ["3", 700e-5],
        ],
        'irf': [['center', 1.3], ['width', 7.8]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}]],
    })
    wanted = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 301e-3],
            ["2", 502e-4],
            ["3", 705e-5],
        ],
        'irf': [['center', 1.3], ['width', 7.8]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}]],
    })

    time = np.arange(-10, 100, 1.5)
    pixel = np.arange(600, 750, 10)
    axis = {"time": time, "pixel": pixel}

    clp = _create_gaussian_clp(
        ['s1', 's2', 's3'], [7, 3, 30], [620, 670, 720], [10, 30, 50], pixel)


class ThreeComponentSequential:
    model = KineticImageModel.from_dict({
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
            'irf1': {'type': 'multi-gaussian', 'center': ['irf.center'], 'width': ['irf.width']},
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })

    initial = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 501e-3],
            ["2", 202e-4],
            ["3", 105e-5],
            {'non-negative': True},
        ],
        'irf': [['center', 1.3], ['width', 7.8]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}],
              ['0', 0, {'vary': False, 'non-negative': False}]],
    })
    wanted = ParameterGroup.from_dict({
        'kinetic': [
            ["1", 501e-3],
            ["2", 202e-4],
            ["3", 105e-5],
        ],
        'irf': [['center', 1.3], ['width', 7.8]],
        'j': [['1', 1, {'vary': False, 'non-negative': False}],
              ['0', 0, {'vary': False, 'non-negative': False}]],
    })

    time = np.asarray(np.arange(-10, 50, 1.0))
    pixel = np.arange(600, 750, 10)
    axis = {"time": time, "pixel": pixel}

    clp = _create_gaussian_clp(
        ['s1', 's2', 's3'], [7, 3, 30], [620, 670, 720], [10, 30, 50], pixel)


@pytest.mark.parametrize("suite", [
    OneComponentOneChannel,
    OneComponentOneChannelGaussianIrf,
    #  OneComponentOneChannelMeasuredIrf,
    ThreeComponentParallel,
    ThreeComponentSequential,
])
@pytest.mark.parametrize("nnls", [True, False])
def test_kinetic_model(suite, nnls):

    model = suite.model
    print(model.validate())
    assert model.valid()

    wanted = suite.wanted
    print(model.validate(wanted))
    print(wanted)
    assert model.valid(wanted)

    initial = suite.initial
    print(model.validate(initial))
    assert model.valid(initial)

    print(model.markdown(initial))

    dataset = model.simulate('dataset1', wanted, suite.axis, suite.clp)

    assert dataset.data.shape == \
        (suite.axis['time'].size, suite.axis['pixel'].size)

    data = {'dataset1': dataset}

    result = model.optimize(initial, data, nnls=nnls, max_nfev=20)
    print(result.optimized_parameter)

    for label, param in result.optimized_parameter.all():
        assert np.allclose(param.value, wanted.get(label).value,
                           rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(dataset['time'], resultdata['time'])
    assert np.array_equal(dataset['pixel'], resultdata['pixel'])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, rtol=1e-2)
    assert 'species_associated_images' in resultdata
    assert 'decay_associated_images' in resultdata

    if len(model.irf) != 0:
        assert 'irf' in resultdata
