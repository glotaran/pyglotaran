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
        mat = list(self.get_model().get_calculated_matrix_group(dataset).groups())[0]
        print(mat)
        return self.estimated_matrix(dataset)

    def concentrations(self, dataset: str) -> np.array:
        """

        Parameters
        ----------
        dataset: str
                Label of the dataset.


        Returns
        -------
        concentrations : numpy.array
            Concentrations

        """
        return self.calculated_matrix(dataset)
