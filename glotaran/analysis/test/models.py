from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr

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
class SimpleTestMegacomplex:
    pass


@model(
    "simple_test",
    attributes={},
    matrix=calculate_c,
    model_dimension="c",
    global_matrix=calculate_e,
    global_dimension="e",
    megacomplex_type=SimpleTestMegacomplex,
)
class SimpleTestModel(Model):
    pass


def calculate_kinetic(dataset_descriptor=None, axis=None, index=None, extra_stuff=None):
    kinpar = -1 * np.asarray(dataset_descriptor.kinetic)
    if dataset_descriptor.label == "dataset3":
        # this case is for the ThreeDatasetDecay test
        compartments = [f"s{i+2}" for i in range(len(kinpar))]
    else:
        compartments = [f"s{i+1}" for i in range(len(kinpar))]
    array = np.exp(np.outer(axis, kinpar))
    return (compartments, array)


def calculate_spectral_simple(dataset_descriptor, axis):
    kinpar = -1 * np.array(dataset_descriptor.kinetic)
    if dataset_descriptor.label == "dataset3":
        # this case is for the ThreeDatasetDecay test
        compartments = [f"s{i+2}" for i in range(len(kinpar))]
    else:
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


def constrain_matrix_function_typecheck(
    model: type[Model],
    label: str,
    parameters: ParameterGroup,
    clp_labels: list[str],
    matrix: np.ndarray,
    index: float,
):
    assert isinstance(label, str)
    assert isinstance(parameters, ParameterGroup)
    assert isinstance(clp_labels, list)
    assert all(isinstance(clp_label, str) for clp_label in clp_labels)
    assert isinstance(matrix, np.ndarray)

    if model.index_dependent():
        assert isinstance(index, float)
    else:
        assert index is None

    model.constrain_matrix_function_called = True

    return (clp_labels, matrix)


def retrieve_clp_typecheck(
    model: type[Model],
    parameters: ParameterGroup,
    clp_labels: dict[str, list[str] | list[list[str]]],
    reduced_clp_labels: dict[str, list[str] | list[list[str]]],
    reduced_clps: dict[str, list[np.ndarray]],
    data: dict[str, xr.Dataset],
) -> dict[str, list[np.ndarray]]:
    assert isinstance(parameters, ParameterGroup)

    assert isinstance(reduced_clps, dict)
    assert all(isinstance(dataset_clps, list) for dataset_clps in reduced_clps.values())

    assert all(
        [isinstance(index_clps, np.ndarray) for index_clps in dataset_clps]
        for dataset_clps in reduced_clps.values()
    )

    assert isinstance(data, dict)
    assert all(isinstance(label, str) for label in data)
    assert all(isinstance(dataset, xr.Dataset) for dataset in data.values())

    assert isinstance(clp_labels, dict)
    assert isinstance(reduced_clp_labels, dict)
    assert all(
        isinstance(dataset_clp_labels, list) for dataset_clp_labels in reduced_clp_labels.values()
    )
    assert all(
        [[isinstance(label, str) for label in index_labels] for index_labels in dataset_clp_labels]
        for dataset_clp_labels in reduced_clp_labels.values()
    )
    if model.index_dependent():
        for dataset_clp_labels in clp_labels.values():
            assert all(isinstance(index_label, list) for index_label in dataset_clp_labels)
            assert all(
                [isinstance(label, str) for label in index_label]
                for index_label in dataset_clp_labels
            )
        assert all(
            [isinstance(index_labels, list) for index_labels in dataset_clp_labels]
            for dataset_clp_labels in reduced_clp_labels.values()
        )

    else:
        for dataset_clp_labels in clp_labels.values():
            assert all(isinstance(label, str) for label in dataset_clp_labels)

    model.retrieve_clp_function_called = True

    return reduced_clps


def additional_penalty_typecheck(
    model: type[Model],
    parameters: ParameterGroup,
    clp_labels: dict[str, list[str] | list[list[str]]],
    clps: dict[str, list[np.ndarray]],
    matrices: dict[str, np.ndarray | list[np.ndarray]],
    data: dict[str, xr.Dataset],
    group_tolerance: float,
) -> np.ndarray:
    assert isinstance(parameters, ParameterGroup)
    assert isinstance(group_tolerance, float)

    assert isinstance(clps, dict)
    assert all(isinstance(dataset_clps, list) for dataset_clps in clps.values())
    assert all(
        [isinstance(index_clps, np.ndarray) for index_clps in dataset_clps]
        for dataset_clps in clps.values()
    )

    assert isinstance(data, dict)
    assert all(isinstance(label, str) for label in data)
    assert all(isinstance(dataset, xr.Dataset) for dataset in data.values())

    assert isinstance(clp_labels, dict)
    assert isinstance(matrices, dict)
    if model.index_dependent():
        for dataset_clp_labels in clp_labels.values():
            assert all(isinstance(index_label, list) for index_label in dataset_clp_labels)
            assert all(
                [isinstance(label, str) for label in index_label]
                for index_label in dataset_clp_labels
            )

        for matrix in matrices.values():
            assert isinstance(matrix, list)
            assert all(isinstance(index_matrix, np.ndarray) for index_matrix in matrix)
    else:
        for dataset_clp_labels in clp_labels.values():
            assert all(isinstance(label, str) for label in dataset_clp_labels)
        for matrix in matrices.values():
            assert isinstance(matrix, np.ndarray)

    model.additional_penalty_function_called = True

    return np.asarray([0.1])


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
        "location": {"type": List[Parameter], "allow_none": True},
        "amplitude": {"type": List[Parameter], "allow_none": True},
        "delta": {"type": List[Parameter], "allow_none": True},
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
    megacomplex_type=SimpleTestMegacomplex,
    has_additional_penalty_function=lambda model: True,
    additional_penalty_function=additional_penalty_typecheck,
    has_matrix_constraints_function=lambda model: True,
    constrain_matrix_function=constrain_matrix_function_typecheck,
    retrieve_clp_function=retrieve_clp_typecheck,
    grouped=lambda model: model.is_grouped,
    index_dependent=lambda model: model.is_index_dependent,
)
class DecayModel(Model):
    additional_penalty_function_called = False
    constrain_matrix_function_called = False
    retrieve_clp_function_called = False
    is_grouped = False
    is_index_dependent = False


@model(
    "multi_channel",
    attributes={},
    dataset_type=GaussianShapeDecayDatasetDescriptor,
    matrix=calculate_kinetic,
    model_dimension="c",
    global_matrix=calculate_spectral_gauss,
    global_dimension="e",
    megacomplex_type=SimpleTestMegacomplex,
    grouped=lambda model: model.is_grouped,
    index_dependent=lambda model: model.is_index_dependent,
    has_additional_penalty_function=lambda model: True,
    additional_penalty_function=additional_penalty_typecheck,
)
class GaussianDecayModel(Model):
    additional_penalty_function_called = False
    constrain_matrix_function_called = False
    retrieve_clp_function_called = False
    is_grouped = False
    is_index_dependent = False


class OneCompartmentDecay:
    scale = 2
    wanted_parameters = ParameterGroup.from_list([101e-4])
    initial_parameters = ParameterGroup.from_list([100e-5, [scale, {"vary": False}]])

    e_axis = np.asarray([1.0])
    c_axis = np.arange(0, 150, 1.5)

    model_dict = {
        "dataset": {
            "dataset1": {"initial_concentration": [], "megacomplex": [], "kinetic": ["1"]}
        },
    }
    sim_model = DecayModel.from_dict(model_dict)
    model_dict["dataset"]["dataset1"]["scale"] = "2"
    model = DecayModel.from_dict(model_dict)


class TwoCompartmentDecay:
    wanted_parameters = ParameterGroup.from_list([11e-4, 22e-5])
    initial_parameters = ParameterGroup.from_list([10e-4, 20e-5])

    e_axis = np.asarray([1.0])
    c_axis = np.arange(0, 150, 1.5)

    model = DecayModel.from_dict(
        {
            "dataset": {
                "dataset1": {"initial_concentration": [], "megacomplex": [], "kinetic": ["1", "2"]}
            },
        }
    )
    sim_model = model


class ThreeDatasetDecay:
    wanted_parameters = ParameterGroup.from_list([101e-4, 201e-3])
    initial_parameters = ParameterGroup.from_list([100e-5, 200e-3])

    e_axis = np.asarray([1.0])
    c_axis = np.arange(0, 150, 1.5)

    e_axis2 = np.asarray([1.0, 2.01])
    c_axis2 = np.arange(0, 100, 1.5)

    e_axis3 = np.asarray([0.99, 3.0])
    c_axis3 = np.arange(0, 150, 1.5)

    model_dict = {
        "dataset": {
            "dataset1": {"initial_concentration": [], "megacomplex": [], "kinetic": ["1"]},
            "dataset2": {"initial_concentration": [], "megacomplex": [], "kinetic": ["1", "2"]},
            "dataset3": {"initial_concentration": [], "megacomplex": [], "kinetic": ["2"]},
        },
    }
    sim_model = DecayModel.from_dict(model_dict)
    model = sim_model


class MultichannelMulticomponentDecay:
    wanted_parameters = ParameterGroup.from_dict(
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
    initial_parameters = ParameterGroup.from_dict({"k": [0.006, 0.003, 0.0003, 0.03]})

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
    model = GaussianDecayModel.from_dict(
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
