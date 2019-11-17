import pytest
import numpy as np

from glotaran.parameter import ParameterGroup
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel


class SimpleIrfDispersion:
    model = KineticSpectrumModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['j.1']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s1", "s1"): 'kinetic.1',
            }}
        },
        'irf': {
            'irf1': {
                'type': 'spectral-gaussian',
                'center': 'irf.center',
                'width': 'irf.width',
                'dispersion_center': 'irf.dispcenter',
                'center_dispersion': ['irf.centerdisp'],
            },
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticSpectrumModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['j.1']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s1", "s1"): 'kinetic.1',
            }}
        },
        'irf': {
            'irf1': {
                'type': 'spectral-gaussian',
                'center': 'irf.center',
                'width': 'irf.width',
                'dispersion_center': 'irf.dispcenter',
                'center_dispersion': ['irf.centerdisp'],
            },
        },
        'shape': {
            'sh1': {
                'type': "one",
            },
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shape': {'s1': 'sh1'}
            },
        },
    })

    initial = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False, 'non-negative': False}],
        ],
        'kinetic': [
            ["1", 0.5],
            {'non-negative': False}
        ],
        'irf': [['center', 0.3],
                ['width', 0.1],
                ['dispcenter', 400, {'vary': False}],
                ['centerdisp', 0.5]],
    })
    wanted = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False, 'non-negative': False}],
        ],
        'kinetic': [
            ["1", 0.5],
        ],

        'irf': [['center', 0.3], ['width', 0.1], ['dispcenter', 400], ['centerdisp', 0.5]],
    })

    time_p1 = np.linspace(-1, 2, 50, endpoint=False)
    time_p2 = np.linspace(2, 5, 30, endpoint=False)
    time_p3 = np.geomspace(5, 10, num=20)
    time = np.concatenate([time_p1, time_p2, time_p3])
    spectral = np.arange(300, 500, 5)
    axis = {"time": time, "spectral": spectral}


class MultiIrfDispersion:
    model = KineticSpectrumModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['j.1']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s1", "s1"): 'kinetic.1',
            }}
        },
        'irf': {
            'irf1': {
                'type': 'spectral-multi-gaussian',
                'center': ['irf.center'],
                'width': ['irf.width'],
                'dispersion_center': 'irf.dispcenter',
                'center_dispersion': ['irf.centerdisp1', 'irf.centerdisp2'],
                'width_dispersion': ['irf.widthdisp'],
            },
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
            },
        },
    })
    sim_model = KineticSpectrumModel.from_dict({
        'initial_concentration': {
            'j1': {
                'compartments': ['s1'],
                'parameters': ['j.1']
            },
        },
        'megacomplex': {
            'mc1': {'k_matrix': ['k1']},
        },
        'k_matrix': {
            "k1": {'matrix': {
                ("s1", "s1"): 'kinetic.1',
            }}
        },
        'irf': {
            'irf1': {
                'type': 'spectral-multi-gaussian',
                'center': ['irf.center'],
                'width': ['irf.width'],
                'dispersion_center': 'irf.dispcenter',
                'center_dispersion': ['irf.centerdisp1', 'irf.centerdisp2'],
                'width_dispersion': ['irf.widthdisp'],
            },
        },
        'shape': {
            'sh1': {
                'type': "one",
            },
        },
        'dataset': {
            'dataset1': {
                'initial_concentration': 'j1',
                'irf': 'irf1',
                'megacomplex': ['mc1'],
                'shape': {'s1': 'sh1'}
            },
        },
    })

    initial = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False, 'non-negative': False}],
        ],
        'kinetic': [
            ["1", 0.5],
            {'non-negative': False}
        ],
        'irf': [['center', 0.3],
                ['width', 0.1],
                ['dispcenter', 400, {'vary': False}],
                ['centerdisp1', 0.01],
                ['centerdisp2', 0.001],
                ['widthdisp', 0.025]],
    })
    wanted = ParameterGroup.from_dict({
        'j': [
            ['1', 1, {'vary': False, 'non-negative': False}],
        ],
        'kinetic': [
            ["1", 0.5],
        ],
        'irf': [['center', 0.3],
                ['width', 0.1],
                ['dispcenter', 400, {'vary': False}],
                ['centerdisp1', 0.01],
                ['centerdisp2', 0.001],
                ['widthdisp', 0.025]],
    })

    time = np.arange(-1, 5, 0.2)
    spectral = np.arange(300, 500, 25)
    axis = {"time": time, "spectral": spectral}


@pytest.mark.parametrize("suite", [
    SimpleIrfDispersion,
    MultiIrfDispersion,
])
def test_spectral_irf(suite):

    model = suite.model
    print(model.validate())
    assert model.valid()

    sim_model = suite.sim_model
    print(sim_model.validate())
    assert sim_model.valid()

    wanted = suite.wanted
    print(sim_model.validate(wanted))
    print(wanted)
    assert sim_model.valid(wanted)

    initial = suite.initial
    print(model.validate(initial))
    assert model.valid(initial)

    print(model.markdown(wanted))

    dataset = sim_model.simulate('dataset1', wanted, suite.axis)

    assert dataset.data.shape == \
        (suite.axis['time'].size, suite.axis['spectral'].size)

    data = {'dataset1': dataset}

    result = model.optimize(initial, data, nnls=True, max_nfev=20)
    print(result.optimized_parameter)

    for label, param in result.optimized_parameter.all():
        assert np.allclose(param.value, wanted.get(label).value,
                           rtol=1e-1)

    resultdata = result.data["dataset1"]

    print(resultdata)

    assert np.array_equal(dataset['time'], resultdata['time'])
    assert np.array_equal(dataset['spectral'], resultdata['spectral'])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    assert np.allclose(dataset.data, resultdata.fitted_data, atol=1e-15)

    print(resultdata.fitted_data.isel(spectral=0).argmax())
    print(resultdata.fitted_data.isel(spectral=-1).argmax())
    assert resultdata.fitted_data.isel(spectral=0).argmax() != \
        resultdata.fitted_data.isel(spectral=-1).argmax()

    assert 'species_associated_spectra' in resultdata
    assert 'decay_associated_spectra' in resultdata
