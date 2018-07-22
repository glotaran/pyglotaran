"""Glotaran Fitmodel Result"""

import numpy as np
from lmfit_varpro import SeparableModelResult

from glotaran.model.dataset import Dataset
from glotaran.model.parameter_group import ParameterGroup


class Result(SeparableModelResult):
    """The result of fit."""

    @property
    def best_fit_parameter(self) -> ParameterGroup:
        """The best fit parameters."""
        return ParameterGroup.from_parameter_dict(self._result.params)

    def estimated_matrix(self, dataset: str) -> np.array:
        """Returns the estimated matrix of the model.

        Parameters
        ----------
        dataset: str
            Label of the dataset.


        Returns
        -------
        e_matrix: np.array
            Estimated Matrix

        """
        return np.asarray(self.e_matrix(**{"dataset": dataset}))

    def calculated_matrix(self, dataset: str) -> np.array:
        """Returns the calculated matrix of the model.

        Parameters
        ----------
        dataset: str
            Label of the dataset.


        Returns
        -------
        c_matrix: np.array
            Calculated Matrix

        """
        return np.asarray(self.c_matrix(**{"dataset": dataset}))

    def fitted_data(self, dataset: str) -> Dataset:
        """Returns the fitted dataset.

        Parameters
        ----------
        dataset: str
            Label of the dataset.


        Returns
        -------
        fitted_dataset: glotaran.Dataset

        """
        data = np.asarray(self.eval(**{"dataset": dataset}))
        dataset = self.model.datasets[dataset].dataset.copy()
        dataset.set(data)
        return dataset

    @property
    def model(self) -> 'glotaran.Model':
        """The Glotaran Model used to fit the data."""
        return self.get_model().model

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
        string += f"{self._result.nfev} |".rjust(lr)
        string += "\n"
        string += "Number of variables |".rjust(ll)
        string += f"{self._result.nvarys} |".rjust(lr)
        string += "\n"
        string += "Number of datapoints |".rjust(ll)
        string += f"{self._result.ndata} |".rjust(lr)
        string += "\n"
        string += "Negrees of freedom |".rjust(ll)
        string += f"{self._result.nfree} |".rjust(lr)
        string += "\n"
        string += "Chi Square |".rjust(ll)
        string += f"{self._result.chisqr:.6f} |".rjust(lr)
        string += "\n"
        string += "Reduced Chi Square |".rjust(ll)
        string += f"{self._result.redchi:.6f} |".rjust(lr)
        string += "\n"

        string += "\n"
        string += "## Best Fit Parameter\n\n"
        string += f"{self.best_fit_parameter}"
        string += "\n"

        return string
