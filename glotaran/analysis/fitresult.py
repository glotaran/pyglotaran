"""This package contains the FitResult object"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from lmfit.minimizer import MinimizerResult


from glotaran.datasets.dataset import Dataset
from glotaran.model.parameter_group import ParameterGroup


@dataclass
class DatasetResult:
    """DatasetResult holds the fitresults for the data."""
    label: str
    compartments: List[str]
    non_linear_matrix: np.ndarray
    conditionally_non_linear_parameter: np.ndarray
    data: Dataset


class FitResult:
    """The result of a fit."""

    def __init__(self,
                 lm_result: MinimizerResult,
                 dataset_results: Dict[str, DatasetResult],
                 ):
        """

        Parameters
        ----------
        lm_result: MinimizerResult
        dataset_results: Dict[str, DatasetResult]

        Returns
        -------
        """
        self._lm_result = lm_result
        self._dataset_results = dataset_results

    @property
    def best_fit_parameter(self) -> ParameterGroup:
        """The best fit parameters."""
        return ParameterGroup.from_parameter_dict(self._lm_result.params)

    def get_dataset(self, label: str):
        """get_dataset returns the DatasetResult for the given dataset.

        Parameters
        ----------
        label : str
            The label of the dataset.

        Returns
        -------
        dataset_result: DatasetResult
            The result for the dataset.
        """
        return self._dataset_results[label]

    def __str__(self):
        string = "# Fitresult\n\n"

        # pylint: disable=invalid-name

        ll = 32
        lr = 13

        string += "Optimization Result".ljust(ll-1)
        string += "|"
        string += "|".rjust(lr)
        string += "\n"
        string += "|".rjust(ll, "-")
        string += "|".rjust(lr, "-")
        string += "\n"

        string += "Number of residual evaluation |".rjust(ll)
        string += f"{self._lm_result.nfev} |".rjust(lr)
        string += "\n"
        string += "Number of variables |".rjust(ll)
        string += f"{self._lm_result.nvarys} |".rjust(lr)
        string += "\n"
        string += "Number of datapoints |".rjust(ll)
        string += f"{self._lm_result.ndata} |".rjust(lr)
        string += "\n"
        string += "Negrees of freedom |".rjust(ll)
        string += f"{self._lm_result.nfree} |".rjust(lr)
        string += "\n"
        string += "Chi Square |".rjust(ll)
        string += f"{self._lm_result.chisqr:.6f} |".rjust(lr)
        string += "\n"
        string += "Reduced Chi Square |".rjust(ll)
        string += f"{self._lm_result.redchi:.6f} |".rjust(lr)
        string += "\n"

        string += "\n"
        string += "## Best Fit Parameter\n\n"
        string += f"{self.best_fit_parameter}"
        string += "\n"

        return string
