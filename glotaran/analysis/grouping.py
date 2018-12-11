"""This package contains functions for creating and calculationg groups."""
import numpy as np

from typing import Dict, Generator, List, Tuple

from glotaran.model.dataset import Dataset
from glotaran.model.dataset_descriptor import DatasetDescriptor
from glotaran.model.parameter_group import ParameterGroup

Group = Dict[any, Tuple[any, DatasetDescriptor]]


def create_group(model,  # temp doc fix : 'glotaran.model.Model',
                 data: Dict[str, Dataset],
                 xtol: float = 0.5,
                 dataset: str = None,
                 ) -> Group:
    """create_group creates a calculation group for a model along the estimated
    axis.

    Parameters
    ----------
    model : glotaran.model.Model
        The model to group.
    data : Dict[str, Dataset]
    xtol : float
        The grouping tolerance.
    dataset : str
        (default = None)
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
            raise Exception("Missing data for dataset '{dataset_descriptor.label}'")
        axis = data[dataset_descriptor.label].get_axis(model.estimated_axis)
        for index in axis:
            if model.allow_grouping:
                group_index = index if not any(_is_close(index, val, xtol) for val in group) \
                    else [val for val in group if _is_close(index, val, xtol)][0]
                if group_index not in group:
                    group[group_index] = []
                group[group_index].append((index, dataset_descriptor))
            else:
                group_index = f'{dataset_descriptor.label}_{index}'
                group[group_index] = [(index, dataset_descriptor)]
    return group


def _is_close(a, b, xtol):
    if np.issubdtype(a, np.number) and np.issubdtype(b, np.number):
        return abs(a - b) < xtol
    return a == b


def calculate_group_item(item,
                         model,  # temp doc fix : 'glotaran.model.Model',
                         parameter: ParameterGroup,
                         data: Dict[str, Dataset],
                         ) -> Generator[Tuple[int, np.ndarray], None, None]:

    if model.calculated_matrix is None:
        raise Exception("Missing function for calculated matrix.")

    full = None
    full_clp = None
    dataset_labels = []
    for index, dataset_descriptor in item:

        if dataset_descriptor.label not in data:
            raise Exception("Missing data for dataset '{dataset_descriptor.label}'")
        dataset_descriptor = dataset_descriptor.fill(model, parameter)

        dataset_labels.append(dataset_descriptor.label)

        axis = data[dataset_descriptor.label].get_axis(model.calculated_axis)

        (clp, this_matrix) = model.calculated_matrix(dataset_descriptor,
                                                     index,
                                                     axis)

        #  apply_constraints(dataset_descriptor, clp, this_matrix, index)

        if full is None:
            full = this_matrix
            full_clp = clp
        else:
            if not clp == full_clp:
                for comp in clp:
                    if comp not in full_clp:
                        full_clp.append(comp)
                        full = np.concatenate((full, np.zeros((full.shape[1]))))
                reshape = np.zeros((len(full_clp), this_matrix.shape[1]))
                for i, comp in enumerate(full_clp):
                    reshape[i, :] = this_matrix[clp.index(comp), :] \
                            if comp in clp else np.zeros((this_matrix.shape[1]))
                this_matrix = reshape

            full = np.concatenate((full, this_matrix), axis=1)

    return (full, full_clp, dataset_labels)


def calculate_group(group: Group,
                    model,  # temp doc fix : 'glotaran.model.Model',
                    parameter: ParameterGroup,
                    data: Dict[str, Dataset],
                    ) -> Generator[Tuple[int, np.ndarray], None, None]:
    """calculate_group calculates a group.

    Parameters
    ----------
    group : Dict[any, Tuple[any, DatasetDescriptor]]
    model : glotaran.model.Model
    parameter : ParameterGroup
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
                      data: Dict[str, Dataset]) -> List[np.ndarray]:
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

    result = []
    for _, item in group.items():
        full = None
        for index, dataset_descriptor in item:

            if dataset_descriptor.label not in data:
                raise Exception("Missing data for dataset '{dataset_descriptor.label}'")

            dataset = data[dataset_descriptor.label]
            axis = list(dataset.get_axis(model.estimated_axis))
            idx = axis.index(index)
            dataset = dataset.data()[idx, :]

            if full is None:
                full = dataset
            else:
                full = np.append(full, dataset)
        result.append(full)
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
