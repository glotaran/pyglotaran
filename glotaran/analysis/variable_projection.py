"""This package contains functions for variable projection."""
from typing import Dict, List
import numpy as np
from scipy.linalg import lapack

from glotaran.model.dataset import Dataset
from glotaran.model.parameter_group import ParameterGroup

from .grouping import Group, calculate_group


def qr_residual(A, B):

    # Kaufman Q2 step 3
    qr, tau, _, _ = lapack.dgeqrf(A)

    # Kaufman Q2 step 4

    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=0)

    for i in range(A.shape[1]):
        B[i] = 0

    # Kaufman Q2 step 5

    B, _, _ = lapack.dormqr("L", "N", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=1)
    return B


def qr_coefficents(A, B):

    # Kaufman Q2 step 3

    qr, tau, _, _ = lapack.dgeqrf(A)

    # Kaufman Q2 step 4

    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]))

    # Kaufman Q2 step 6

    P, _ = lapack.dtrtrs(qr, B)
    return P


#  def residual_variable_projection(parameter: ParameterGroup,
#                                   group: Group,
#                                   model,  # temp doc fix : 'glotaran.model.BaseModel',
#                                   data: Dict[str, Dataset],
#                                   data_group: List[np.ndarray],
#                                   **kwargs) -> np.ndarray:
#      """residul_variable_projection returns the variable projection residual.
#
#      Parameters
#      ----------
#      parameter : ParameterGroup
#      group: Dict[any, Tuple[any, DatasetDescriptor]]
#      model : glotaran.model.BaseModel
#      data : Dict[str, Dataset]
#          A dictionary of dataset labels and Datasets.
#      data_group : List[np.ndarray]
#      **kwargs
#
#      Returns
#      -------
#      residual: np.ndarray
#      """
#      res = np.concatenate([qr_residual(matrix.T, data_group[i]) for i, matrix, _, _ in
#                            calculate_group(group, model, parameter, data)])
#      return res


def residual_variable_projection(matrix, data) -> np.ndarray:
    """residul_variable_projection returns the variable projection residual.

    Parameters
    ----------
    parameter : ParameterGroup
    group: Dict[any, Tuple[any, DatasetDescriptor]]
    model : glotaran.model.BaseModel
    data : Dict[str, Dataset]
        A dictionary of dataset labels and Datasets.
    data_group : List[np.ndarray]
    **kwargs

    Returns
    -------
    residual: np.ndarray
    """
    return qr_residual(matrix.T, data)


def clp_variable_projection(parameter: ParameterGroup,
                            group: Group,
                            model,  # temp doc fix : 'glotaran.model.BaseModel',
                            data: Dict[str, Dataset],
                            data_group: List[np.ndarray],
                            ) -> np.ndarray:
    """residul_variable_projection returns the variable projection residual.

    Parameters
    ----------
    parameter : ParameterGroup
    group: Dict[any, Tuple[any, DatasetDescriptor]]
    model : glotaran.model.BaseModel
    data : Dict[str, Dataset]
        A dictionary of dataset labels and Datasets.
    data_group : List[np.ndarray]
    **kwargs

    Returns
    -------
    residual: np.ndarray
    """
    res = {}
    for i, matrix, clp, datasets in calculate_group(group, model, parameter, data):
        res[i] = (datasets, clp, qr_coefficents(matrix.T,
                                                data_group[i])[:len(clp)])
    return res
