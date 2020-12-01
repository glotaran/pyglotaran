from typing import List

import numpy as np

from glotaran.model import DatasetDescriptor
from glotaran.model import Model
from glotaran.model import model
from glotaran.model import model_attribute
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup


def calculate_c(dataset_descriptor=None, axis=None, index=None):
    compartments = ["s1", "s2"]
    r_compartments = []
    array = np.zeros((axis.shape[0], len(compartments)))

    for i in range(len(compartments)):
        r_compartments.append(compartments[i])
        for j in range(axis.shape[0]):
            array[j, i] = (i + j) * axis[j]
    return (r_compartments, array)


def calculate_e(dataset, axis):
    compartments = ["s1", "s2"]
    r_compartments = []
    array = np.zeros((axis.shape[0], len(compartments)))

    for i in range(len(compartments)):
        r_compartments.append(compartments[i])
        for j in range(axis.shape[0]):
            array[j, i] = (i + j) * axis[j]
    return (r_compartments, array)


@model_attribute(
    properties={
        "grouped": bool,
        "indexdependent": bool,
    }
)
class MockMegacomplex:
    pass


@model(
    "mock",
    attributes={},
    matrix=calculate_c,
    model_dimension="c",
    global_matrix=calculate_e,
    global_dimension="e",
    megacomplex_type=MockMegacomplex,
)
class MockModel(Model):
    pass


def calculate_kinetic(dataset_descriptor=None, axis=None, index=None, extra_stuff=None):
    kinpar = -1 * np.array(dataset_descriptor.kinetic)
    compartments = [f"s{i+1}" for i in range(len(kinpar))]
    array = np.exp(np.outer(axis, kinpar))
    return (compartments, array)


def calculate_spectral_simple(dataset, axis):
    kinpar = -1 * np.array(dataset.kinetic)
    compartments = [f"s{i+1}" for i in range(len(kinpar))]
    array = np.asarray([[1 for _ in range(axis.size)] for _ in compartments])
    return compartments, array.T


def calculate_spectral_gauss(dataset, axis):
    location = np.asarray(dataset.location)
    amp = np.asarray(dataset.amplitude)
    delta = np.asarray(dataset.delta)

    array = np.empty((location.size, axis.size), dtype=np.float64)

    for i in range(location.size):
        array[i, :] = amp[i] * np.exp(-np.log(2) * np.square(2 * (axis - location[i]) / delta[i]))
    compartments = [f"s{i+1}" for i in range(location.size)]
    return compartments, array.T


@model_attribute(
    properties={
        "kinetic": List[Parameter],
    }
)
class DecayDatasetDescriptor(DatasetDescriptor):
    pass


@model_attribute(
    properties={
        "kinetic": List[Parameter],
        "location": List[Parameter],
        "amplitude": List[Parameter],
        "delta": List[Parameter],
    }
)
class GaussianShapeDecayDatasetDescriptor(DatasetDescriptor):
    pass


@model(
    "one_channel",
    attributes={},
    dataset_type=DecayDatasetDescriptor,
    matrix=calculate_kinetic,
    model_dimension="c",
    global_matrix=calculate_spectral_simple,
    global_dimension="e",
    megacomplex_type=MockMegacomplex,
    additional_penalty_function=lambda model, parameter, clp_labels, clps, index: [],
)
class DecayModel(Model):
    pass


@model(
    "multi_channel",
    attributes={},
    dataset_type=GaussianShapeDecayDatasetDescriptor,
    matrix=calculate_kinetic,
    model_dimension="c",
    global_matrix=calculate_spectral_gauss,
    global_dimension="e",
    megacomplex_type=MockMegacomplex,
)
class GaussianDecayModel(Model):
    pass


class OneCompartmentDecay:
    wanted = ParameterGroup.from_list([101e-4])
    initial = ParameterGroup.from_list([100e-5])

    e_axis = np.asarray([1])
    c_axis = np.arange(0, 150, 1.5)

    model = DecayModel.from_dict(
        {
            "compartment": ["s1"],
            "dataset": {
                "dataset1": {"initial_concentration": [], "megacomplex": [], "kinetic": ["1"]}
            },
        }
    )
    sim_model = model


class TwoCompartmentDecay:
    wanted = ParameterGroup.from_list([11e-4, 22e-5])
    initial = ParameterGroup.from_list([10e-4, 20e-5])

    e_axis = np.asarray([1])
    c_axis = np.arange(0, 150, 1.5)

    model = DecayModel.from_dict(
        {
            "compartment": ["s1", "s2"],
            "dataset": {
                "dataset1": {"initial_concentration": [], "megacomplex": [], "kinetic": ["1", "2"]}
            },
        }
    )
    sim_model = model


class MultichannelMulticomponentDecay:
    wanted = ParameterGroup.from_dict(
        {
            "k": [0.006, 0.003, 0.0003, 0.03],
            "loc": [
                ["1", 14705],
                ["2", 13513],
                ["3", 14492],
                ["4", 14388],
            ],
            "amp": [
                ["1", 1],
                ["2", 2],
                ["3", 5],
                ["4", 20],
            ],
            "del": [
                ["1", 400],
                ["2", 100],
                ["3", 300],
                ["4", 200],
            ],
        }
    )
    initial = ParameterGroup.from_dict({"k": [0.006, 0.003, 0.0003, 0.03]})

    e_axis = np.arange(12820, 15120, 50)
    c_axis = np.arange(0, 150, 1.5)

    sim_model = GaussianDecayModel.from_dict(
        {
            "compartment": ["s1", "s2", "s3", "s4"],
            "dataset": {
                "dataset1": {
                    "initial_concentration": [],
                    "megacomplex": [],
                    "kinetic": ["k.1", "k.2", "k.3", "k.4"],
                    "location": ["loc.1", "loc.2", "loc.3", "loc.4"],
                    "delta": ["del.1", "del.2", "del.3", "del.4"],
                    "amplitude": ["amp.1", "amp.2", "amp.3", "amp.4"],
                }
            },
        }
    )
    model = DecayModel.from_dict(
        {
            "compartment": ["s1", "s2", "s3", "s4"],
            "dataset": {
                "dataset1": {
                    "initial_concentration": [],
                    "megacomplex": [],
                    "kinetic": ["k.1", "k.2", "k.3", "k.4"],
                }
            },
        }
    )
