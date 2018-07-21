"""Glotaran Kinetic Result"""
import numpy as np

from glotaran.fitmodel import Result


class KineticResult(Result):
    """The result of kinetic model fit"""

    def decay_associated_spectra(self, dataset: str) -> np.array:
        """

        Parameters
        ----------
        dataset: str
            Label of the dataset.


        Returns
        -------
        das : numpy.array
            Decay Associated Spectra
        """
        return self.estimated_matrix(dataset)

    def concentraions(self, dataset: str) -> np.array:
        """

        Parameters
        ----------
        dataset: str
                Label of the dataset.


        Returns
        -------
        concentraions : numpy.array
            Concentrations

        """
        return self.calculated_matrix(dataset)
