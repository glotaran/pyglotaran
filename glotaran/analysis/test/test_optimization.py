import pytest
from typing import List
import numpy as np

from glotaran.analysis.simulation import simulate
from glotaran.analysis.scheme import Scheme
from glotaran.analysis.optimize import optimize

from glotaran.model import DatasetDescriptor, Model, model_attribute, model
from glotaran.parameter import Parameter, ParameterGroup

from .mock import MockMegacomplex


def calculate_kinetic(dataset_descriptor=None, axis=None, index=None, extra_stuff=None):
    kinpar = -1 * np.array(dataset_descriptor.kinetic)
    compartments = [f's{i+1}' for i in range(len(kinpar))]
    array = np.exp(np.outer(axis, kinpar))
    return (compartments, array)


def calculate_spectral_simple(dataset, axis):
    kinpar = -1 * np.array(dataset.kinetic)
    compartments = [f's{i+1}' for i in range(len(kinpar))]
    array = np.asarray([[1 for _ in range(axis.size)] for _ in compartments])
    return compartments, array.T


def calculate_spectral_gauss(dataset, axis):
    location = np.asarray(dataset.location)
    amp = np.asarray(dataset.amplitude)
    delta = np.asarray(dataset.delta)

    array = np.empty((location.size, axis.size), dtype=np.float64)

    for i in range(location.size):
        array[i, :] = amp[i] * np.exp(
            -np.log(2) * np.square(
                2 * (axis - location[i])/delta[i]
            )
        )
    compartments = [f's{i+1}' for i in range(location.size)]
    return compartments, array.T


@model_attribute(properties={
    'kinetic': List[Parameter],
})
class DecayDatasetDescriptor(DatasetDescriptor):
    pass


@model_attribute(properties={
    'kinetic': List[Parameter],
    'location': List[Parameter],
    'amplitude': List[Parameter],
    'delta': List[Parameter],
})
class GaussianShapeDecayDatasetDescriptor(DatasetDescriptor):
    pass


@model('one_channel',
       dataset_type=DecayDatasetDescriptor,
       matrix=calculate_kinetic,
       matrix_dimension='c',
       global_matrix=calculate_spectral_simple,
       global_dimension='e',
       megacomplex_type=MockMegacomplex,
       additional_penalty_function=lambda model, parameter, clp_labels, clps, index: [],
       )
class DecayModel(Model):
    pass


@model('multi_channel',
       dataset_type=GaussianShapeDecayDatasetDescriptor,
       matrix=calculate_kinetic,
       matrix_dimension='c',
       global_matrix=calculate_spectral_gauss,
       global_dimension='e',
       megacomplex_type=MockMegacomplex,
       )
class GaussianDecayModel(Model):
    pass


class OneCompartmentDecay:
    wanted = ParameterGroup.from_list([101e-4])
    initial = ParameterGroup.from_list([100e-5])

    e_axis = np.asarray([1])
    c_axis = np.arange(0, 150, 1.5)

    model = DecayModel.from_dict({
        'compartment': ["s1"],
        'dataset': {
            'dataset1': {
                'initial_concentration': [],
                'megacomplex': [],
                'kinetic': ['1']
            }
        }
    })
    sim_model = model


class TwoCompartmentDecay:
    wanted = ParameterGroup.from_list([11e-4, 22e-5])
    initial = ParameterGroup.from_list([10e-4, 20e-5])

    e_axis = np.asarray([1])
    c_axis = np.arange(0, 150, 1.5)

    model = DecayModel.from_dict({
        'compartment': ["s1", "s2"],
        'dataset': {
            'dataset1': {
                'initial_concentration': [],
                'megacomplex': [],
                'kinetic': ['1', '2']
            }
        }
    })
    sim_model = model


class MultichannelMulticomponentDecay:
    wanted = ParameterGroup.from_dict({
        'k': [.006, 0.003, 0.0003, 0.03],
        'loc': [
            ['1', 14705],
            ['2', 13513],
            ['3', 14492],
            ['4', 14388],
        ],
        'amp': [
            ['1', 1],
            ['2', 2],
            ['3', 5],
            ['4', 20],
        ],
        'del': [
            ['1', 400],
            ['2', 100],
            ['3', 300],
            ['4', 200],
        ]
    })
    initial = ParameterGroup.from_dict({'k': [.006, 0.003, 0.0003, 0.03]})

    e_axis = np.arange(12820, 15120, 50)
    c_axis = np.arange(0, 150, 1.5)

    sim_model = GaussianDecayModel.from_dict({
        'compartment': ["s1", "s2", "s3", "s4"],
        'dataset': {
            'dataset1': {
                'initial_concentration': [],
                'megacomplex': [],
                'kinetic': ['k.1', 'k.2', 'k.3', 'k.4'],
                'location': ['loc.1', 'loc.2', 'loc.3', 'loc.4'],
                'delta': ['del.1', 'del.2', 'del.3', 'del.4'],
                'amplitude': ['amp.1', 'amp.2', 'amp.3', 'amp.4'],
            }
        }
    })
    model = DecayModel.from_dict({
        'compartment': ["s1", "s2", "s3", "s4"],
        'dataset': {
            'dataset1': {
                'initial_concentration': [],
                'megacomplex': [],
                'kinetic': ['k.1', 'k.2', 'k.3', 'k.4']
            }
        }
    })


@pytest.mark.parametrize("index_dependend", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("suite", [
    OneCompartmentDecay,
    TwoCompartmentDecay,
    MultichannelMulticomponentDecay
])
def test_fitting(suite, index_dependend, grouped):
    model = suite.model

    def gr():
        return grouped
    model.grouped = gr

    def id():
        return index_dependend
    model.index_dependend = id

    sim_model = suite.sim_model
    est_axis = suite.e_axis
    cal_axis = suite.c_axis

    print(model.validate())
    assert model.valid()

    print(sim_model.validate())
    assert sim_model.valid()

    wanted = suite.wanted
    print(wanted)
    print(sim_model.validate(wanted))
    assert sim_model.valid(wanted)

    initial = suite.initial
    print(initial)
    print(model.validate(initial))
    assert model.valid(initial)

    dataset = simulate(sim_model, 'dataset1', wanted, {'e': est_axis, 'c': cal_axis})
    print(dataset)

    assert dataset.data.shape == (cal_axis.size, est_axis.size)

    data = {'dataset1': dataset}
    scheme = Scheme(model=model, parameter=initial, data=data, nfev=5)

    result = optimize(scheme)
    print(result.optimized_parameter)
    print(result.data['dataset1'])

    for _, param in result.optimized_parameter.all():
        assert np.allclose(param.value, wanted.get(param.full_label).value,
                           rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(dataset.c, resultdata.c)
    assert np.array_equal(dataset.e, resultdata.e)
    assert dataset.data.shape == resultdata.data.shape
    print(dataset.data[0, 0], resultdata.data[0, 0])
    assert np.allclose(dataset.data, resultdata.data)
