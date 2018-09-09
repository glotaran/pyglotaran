"""Glotaran Fitmodel Result"""

import numpy as np
from dataclasses import dataclass
from typing import List


from glotaran.model.dataset import Dataset
from glotaran.model.parameter_group import ParameterGroup


class FitResult:
    """The result of fit."""
    def __init__(self,
                 lm_result,
                 dataset_results,
                 ):
        self._lm_result = lm_result
        self._dataset_results = dataset_results

    @property
    def best_fit_parameter(self) -> ParameterGroup:
        """The best fit parameters."""
        return ParameterGroup.from_parameter_dict(self._lm_result.params)

    def get_dataset(self, label: str):
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


@dataclass
class DatasetResult:
    label: str
    compartments: List[str]
    calculated_matrix: np.ndarray
    estimated_matrix: np.ndarray
    data: Dataset
