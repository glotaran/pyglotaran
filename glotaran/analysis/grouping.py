"""This package contains functions for creating and calculationg groups."""

from typing import Dict, Generator, List, Tuple, Union

import numpy as np
import xarray as xr

from glotaran.model.dataset_descriptor import DatasetDescriptor
from glotaran.parameter import ParameterGroup

Group = Dict[any, Tuple[any, DatasetDescriptor]]


def create_group(model,  # temp doc fix : 'glotaran.model.Model',
                 data: Dict[str, Union[xr.Dataset, xr.DataArray]],
                 atol: float = 0.0,
                 dataset: str = None,
                 ) -> Group:
    """create_group creates a calculation group for a model along the estimated
    axis.

    Parameters
    ----------
    model : glotaran.model.Model
        The model to group.
    data : Dict[str, Dataset]
    atol : float (default = 0.0)
        The grouping tolerance.
    dataset : str (default = None)
        If not None, the group will be created only for the given dataset.

    Returns
    -------
    group : dict(any, tuple()any, DatasetDescriptor))
    """
    group = {}

    for _, dataset_descriptor in model.dataset.items():
        if dataset is not None and not dataset_descriptor.label == dataset:
            continue
        if dataset_descriptor.label not in data:
            raise Exception(f"Missing data for dataset '{dataset_descriptor.label}'")
        axis = data[dataset_descriptor.label][model.estimated_axis].values
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


def _is_close(a, b, atol):
    try:
        return np.all(np.isclose(a, b, atol=atol))
    except Exception:
        return False


def calculate_group_item(item,
                         model,  # temp doc fix : 'glotaran.model.Model',
                         parameter: ParameterGroup,
                         data: Dict[str, Union[xr.Dataset, xr.DataArray]],
                         ) -> Generator[Tuple[int, np.ndarray], None, None]:

    if model.calculated_matrix is None:
        raise Exception("Missing function for calculated matrix.")

    full_clp = None
    full_matrix = None
    for index, dataset_descriptor in item:

        if dataset_descriptor.label not in data:
            raise Exception("Missing data for dataset '{dataset_descriptor.label}'")
        dataset_descriptor = dataset_descriptor.fill(model, parameter)

        dataset = data[dataset_descriptor.label]
        axis = dataset.coords[model.calculated_axis].values

        (clp, matrix) = model.calculated_matrix(dataset_descriptor, index, axis)

        if 'concentration' not in dataset:
            dataset.coords['clp_label'] = clp
            dataset['concentration'] = (
                (
                    model.estimated_axis,
                    model.calculated_axis,
                    'clp_label',
                ),
                np.zeros((
                    dataset.coords[model.estimated_axis].size,
                    axis.size,
                    len(clp),
                ), dtype=np.float64))
        dataset.concentration.loc[{model.estimated_axis: index}] = matrix

        if 'weight' in dataset:
            for i in range(matrix.shape[1]):
                matrix[:, i] = np.multiply(
                    matrix[:, i], dataset.weight.sel({model.estimated_axis, index})
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

    if callable(model._constrain_calculated_matrix_function):
        (full_clp, full_matrix) = \
            model._constrain_calculated_matrix_function(parameter, full_clp, full_matrix, index)

    return (full_clp, full_matrix)


def calculate_group(group: Group,
                    model,  # temp doc fix : 'glotaran.model.Model',
                    parameter: ParameterGroup,
                    data: Dict[str, Union[xr.Dataset, xr.DataArray]],
                    ) -> Generator[Tuple[int, np.ndarray], None, None]:
    """calculate_group calculates a group.

    Parameters
    ----------
    group : Dict[any, Tuple[any, DatasetDescriptor]]
    model : glotaran.model.Model
    parameter : glotaran.model.ParameterGroup
    data : Dict[str, Dataset]

    Yields
    ------
    (index, array) : tuple(int, np.ndarray)
    """

    i = 0

    for _, item in group.items():

        yield (i,) + calculate_group_item(item, model, parameter, data)
        i += 1


def create_data_group(model,  # temp doc fix : 'glotaran.model.Model',
                      group: Group,
                      data: Dict[str, Union[xr.Dataset, xr.DataArray]],
                      ) -> List[np.ndarray]:
    """create_data_group returns the datagroup for the model.

    Parameters
    ----------
    model : glotaran.model.Model
    group : dict(any, tuple(any, DatasetDescriptor))
    data : Dict[str, Dataset])

    Returns
    -------
    datagroup : list(np.ndarray)
    """

    result = {}
    for i, item in group.items():
        full = None
        for index, dataset_descriptor in item:

            if dataset_descriptor.label not in data:
                raise Exception("Missing data for dataset '{dataset_descriptor.label}'")

            dataset = data[dataset_descriptor.label]
            dataset = dataset.weighted_data if 'weighted_data' in dataset else dataset.data
            dataset = dataset.sel({model.estimated_axis: index}).values
            if full is None:
                full = dataset
            else:
                full = np.append(full, dataset)
        result[i] = full
    return result


def apply_constraints(dataset, compartments: List[str], matrix: np.ndarray, index):
    if dataset.compartment_constraints is None:
        return

    for constraint in dataset.compartment_constraints:

        if not constraint.applies(index) or constraint.type == 'equal_area':
            continue

        idx = compartments.index(constraint.compartment)
        matrix[idx, :].fill(0.0)
        if constraint.type == 'equal':
            for target, param in constraint.targets.items():
                t_idx = compartments.index(target)
                matrix[idx, :] += param * matrix[t_idx, :]
