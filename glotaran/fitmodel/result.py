"""Glotaran Fitmodel Result"""

import numpy as np
from lmfit_varpro import SeparableModelResult

from glotaran import Dataset, Model


class Result(SeparableModelResult):
    """The result of fit."""

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
    def model(self) -> Model:
        """ """
        return self.get_model().model
