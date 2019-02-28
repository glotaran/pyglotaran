"""Functions for creating and calculating global analysis groups."""

import typing

import numpy as np
import xarray as xr

import glotaran
from glotaran.model.dataset_descriptor import DatasetDescriptor
from glotaran.parameter import ParameterGroup

Group = typing.Dict[typing.Any, typing.List[typing.Tuple[typing.Any, DatasetDescriptor]]]
"""A global analysis group is a dictonary which keys are indices in the global dimension and its
values are `GroupItem`s"""

GroupItem = typing.List[typing.Tuple[typing.Any, DatasetDescriptor]]
"""A global analysis group item is a list of tuples containing an indix on the global dimension and
a :class:`glotaran.model.DatasetDescriptor`"""


def create_group(model: 'glotaran.model.Model',
                 data: typing.Dict[str, xr.Dataset],
                 atol: float = 0.0,
                 ) -> Group:
    """Creates a global analysis group for a model along the global dimension.

    Parameters
    ----------
    model :
        The global analysis model.
    data :
        The data to analyze.
    atol :
        The grouping tolerance.
    """

    def _is_close(a, b, atol):
        try:
            return np.all(np.isclose(a, b, atol=atol))
        except Exception:
            return False

    group = {}
    for dataset_descriptor in model.dataset.values():
        if dataset_descriptor.label not in data:
            raise Exception(f"Missing data for dataset '{dataset_descriptor.label}'")
        axis = data[dataset_descriptor.label][model.global_dimension].values
        for index in axis:
            if model._allow_grouping:
                group_index = index if not any(_is_close(index, val, atol) for val in group) \
                    else [val for val in group if _is_close(index, val, atol)][0]
                if group_index not in group:
                    group[group_index] = []
                group[group_index].append((index, dataset_descriptor))
            else:
                group_index = f'{dataset_descriptor.label}_{index}'
                group[group_index] = [(index, dataset_descriptor)]
    return group


def calculate_group_item(item: GroupItem,
                         model: 'glotaran.model.Model',
                         parameter: ParameterGroup,
                         data: typing.Dict[str, xr.Dataset],
                         ) -> typing.Tuple[typing.List[str], np.ndarray]:
    """Calculates the matrix for the group item and returns a Tuple containing a list of
    conditionaly linear parameters and he resulting matrix.

    Parameters
    ----------
    item :
        The item to calculate.
    parameter :
        The parameter for the calculation.
    data : typing.Dict[str, xr.Dataset]
        The data to analyze.
    """

    if model.matrix is None:
        raise Exception("Missing function for calculating the model matrix.")

    full_clp = None
    full_matrix = None
    for index, dataset_descriptor in item:

        if dataset_descriptor.label not in data:
            raise Exception("Missing data for dataset '{dataset_descriptor.label}'")
        dataset_descriptor = dataset_descriptor.fill(model, parameter)

        dataset = data[dataset_descriptor.label]
        axis = dataset.coords[model.matrix_dimension].values

        (clp, matrix) = model.matrix(dataset_descriptor, dataset, index)

        if 'concentration' not in dataset:
            dataset.coords['clp_label'] = clp
            dataset['concentration'] = (
                (
                    model.global_dimension,
                    model.matrix_dimension,
                    'clp_label',
                ),
                np.zeros((
                    dataset.coords[model.global_dimension].size,
                    axis.size,
                    len(clp),
                ), dtype=np.float64))
        dataset.concentration.loc[{model.global_dimension: index}] = matrix

        if 'weight' in dataset:
            for i in range(matrix.shape[1]):
                matrix[:, i] = np.multiply(
                    matrix[:, i], dataset.weight.sel({model.global_dimension, index})
                )

        if dataset_descriptor.scale:
            matrix *= dataset_descriptor.scale

        if full_matrix is None:
            full_matrix = matrix
            full_clp = clp
        else:
            if not clp == full_clp:
                for comp in clp:
                    if comp not in full_clp:
                        full_clp.append(comp)
                        full_matrix = np.concatenate(
                            (full_matrix, np.zeros((full_matrix.shape[0], 1))), axis=1)
                reshape = np.zeros((matrix.shape[0], len(full_clp)))
                for i, comp in enumerate(full_clp):
                    reshape[:, i] = matrix[:, clp.index(comp)] \
                            if comp in clp else np.zeros((matrix.shape[0]))
                matrix = reshape

            full_matrix = np.concatenate([full_matrix, matrix], axis=0)

    # Apply constraints

    if callable(model._constrain_matrix_function):
        (full_clp, full_matrix) = \
            model._constrain_matrix_function(parameter, full_clp, full_matrix, index)

    return (full_clp, full_matrix)


def create_data_group(model: 'glotaran.model.Model',
                      group: Group,
                      data: typing.Dict[str, xr.Dataset],
                      ) -> typing.List[np.ndarray]:
    """Creates a group of data for global analysis.

    Parameters
    ----------
    model :
        The global analysis model.
    group :
        The analysis group to create the datagroup for.
    data :
        The data to analyze.
    """

    result = {}
    for i, item in group.items():
        full = None
        for index, dataset_descriptor in item:

            if dataset_descriptor.label not in data:
                raise Exception("Missing data for dataset '{dataset_descriptor.label}'")

            dataset = data[dataset_descriptor.label]
            dataset = dataset.weighted_data if 'weighted_data' in dataset else dataset.data
            dataset = dataset.sel({model.global_dimension: index}).values
            if full is None:
                full = dataset
            else:
                full = np.append(full, dataset)
        result[i] = full
    return result
