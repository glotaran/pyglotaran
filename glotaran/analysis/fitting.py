"""This package contains the fit functions."""

from typing import Dict

from glotaran.model.dataset import Dataset
from ..model.parameter_group import ParameterGroup

from .fitresult import FitResult


def fit(model,  # temp doc fix : 'glotaran.model.BaseModel',
        parameter: ParameterGroup,
        data: Dict[str, Dataset],
        verbose: int = 2,
        max_nfev: int = None,) -> FitResult:
    """fit performs a fit of the model.

    Parameters
    ----------
    model : glotaran.model.BaseModel
        The Model to fit.
    parameter : ParameterGroup
        The initial fit parameter.
    data : Dict[str, Dataset],
        A dictionary of dataset labels and Datasets.
    verbose : int
        (optional default=2)
        Set 0 for no log output, 1 for only result and 2 for full verbosity
    max_fnev : int
        (optional default=None)
        The maximum number of function evaluations, default None.

    Returns
    -------
    result: FitResult
        The result of the fit.
    """

    result = FitResult(model, data, parameter, False)
    result.minimize(verbose=verbose, max_nfev=max_nfev)
    return result
