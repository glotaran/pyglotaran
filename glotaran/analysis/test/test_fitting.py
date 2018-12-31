import pytest
from typing import List
import numpy as np

from glotaran.analysis.simulation import simulate
from glotaran.analysis.fitresult import FitResult

from ...model import DatasetDescriptor, BaseModel, ParameterGroup, model_item, model

from .mock import MockMegacomplex

OLD_MAT = None
OLD_KIN = []


def calculate_kinetic(dataset, index, axis):
    global OLD_MAT
    global OLD_KIN
    kinpar = -1 * np.array(dataset.kinetic)
    compartments = [f's{i+1}' for i in range(len(kinpar))]
    if np.array_equal(kinpar, OLD_KIN):
        return (compartments, OLD_MAT)

    array = np.exp(np.outer(axis, kinpar))
    OLD_KIN = kinpar
    OLD_MAT = array
    return (compartments, array)


def calculate_spectral_simple(dataset, axis):
    kinpar = -1 * np.array(dataset.kinetic)
    compartments = [f's{i+1}' for i in range(len(kinpar))]
    array = np.asarray([[1 for _ in range(axis.size)] for _ in compartments])
    return array


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
    return array


@model_item(attributes={
    'kinetic': List[str],
})
class DecayDatasetDescriptor(DatasetDescriptor):
    pass


@model_item(attributes={
    'kinetic': List[str],
    'location': List[str],
    'amplitude': List[str],
    'delta': List[str],
})
class GaussianShapeDecayDatasetDescriptor(DatasetDescriptor):
    pass


@model('one_channel',
       dataset_type=DecayDatasetDescriptor,
       calculated_matrix=calculate_kinetic,
       calculated_axis='c',
       estimated_matrix=calculate_spectral_simple,
       estimated_axis='e',
       megacomplex_type=MockMegacomplex,
       )
class DecayModel(BaseModel):
    pass


@model('multi_channel',
       dataset_type=GaussianShapeDecayDatasetDescriptor,
       calculated_matrix=calculate_kinetic,
       calculated_axis='c',
       estimated_matrix=calculate_spectral_gauss,
       estimated_axis='e',
       megacomplex_type=MockMegacomplex,
       )
class GaussianDecayModel(BaseModel):
    pass


class OneCompartmentDecay:
    wanted = [101e-4]
    initial = [100e-5]

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
    wanted = [101e-4, 202e-5]
    initial = [100e-5, 400e-6]

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
    wanted = [.006667, 0.00333, 0.00035, 0.0303, 0.000909,
              {'loc': [
                  ['1', 14705],
                  ['2', 13513],
                  ['3', 14492],
                  ['4', 14388],
                  ['5', 14184],
              ]},
              {'amp': [
                  ['1', 1],
                  ['2', 2],
                  ['3', 5],
                  ['4', 20],
                  ['5', 10],
              ]},
              {'del': [
                  ['1', 400],
                  ['2', 1000],
                  ['3', 300],
                  ['4', 200],
                  ['5', 350],
              ]},
              ]
    initial = [.005, 0.003, 0.00022, 0.0300, 0.000888]

    e_axis = np.arange(12820, 15120, 4.6)
    c_axis = np.arange(0, 150, 1.5)

    sim_model = GaussianDecayModel.from_dict({
        'compartment': ["s1", "s2", "s3", "s4", "s5"],
        'dataset': {
            'dataset1': {
                'initial_concentration': [],
                'megacomplex': [],
                'kinetic': ['1', '2', '3', '4', '5'],
                'location': ['loc.1', 'loc.2', 'loc.3', 'loc.4', 'loc.5'],
                'delta': ['del.1', 'del.2', 'del.3', 'del.4', 'del.5'],
                'amplitude': ['amp.1', 'amp.2', 'amp.3', 'amp.4', 'amp.5'],
            }
        }
    })
    model = DecayModel.from_dict({
        'compartment': ["s1", "s2", "s3", "s4", "s5"],
        'dataset': {
            'dataset1': {
                'initial_concentration': [],
                'megacomplex': [],
                'kinetic': ['1', '2', '3', '4', '5']
            }
        }
    })


@pytest.mark.parametrize("suite", [
    OneCompartmentDecay,
    TwoCompartmentDecay,
    MultichannelMulticomponentDecay
])
def test_fitting(suite):
    model = suite.model
    sim_model = suite.sim_model
    est_axis = suite.e_axis
    cal_axis = suite.c_axis

    print(model.errors())
    assert model.valid()

    print(sim_model.errors())
    assert sim_model.valid()

    wanted = ParameterGroup.from_list(suite.wanted)
    print(sim_model.errors_parameter(wanted))
    assert sim_model.valid_parameter(wanted)

    initial = ParameterGroup.from_list(suite.initial)
    print(model.errors_parameter(initial))
    assert model.valid_parameter(initial)

    dataset = simulate(sim_model, wanted, 'dataset1', {'e': est_axis, 'c': cal_axis})

    assert dataset.data().shape == (cal_axis.size, est_axis.size)

    data = {'dataset1': dataset}

    result = FitResult(model, data, initial, False)
    result.minimize()
    print(result.best_fit_parameter)

    for param in result.best_fit_parameter.all():
        assert np.allclose(param.value, wanted.get(param.label).value,
                           rtol=1e-1)

    resultdata = result.get_fitted_dataset("dataset1")
    assert np.array_equal(dataset.get_axis('c'), resultdata.get_axis('c'))
    assert np.array_equal(dataset.get_axis('e'), resultdata.get_axis('e'))
    assert dataset.data().shape == resultdata.data().shape
    print(dataset.data()[0, 0], resultdata.data()[0, 0])
    assert np.allclose(dataset.data(), resultdata.data())


if __name__ == '__main__':
    pytest.main(__file__)
