"""Glotaran Kinetic Result"""
import numpy as np

from glotaran.analysis.fitresult import FitResult


class KineticResult(FitResult):
    """The result of kinetic model fit"""

    def species_associated_spectra(self, dataset: str) -> np.ndarray:
        """

        Parameters
        ----------
        dataset: str
            Label of the dataset.


        Returns
        -------
        das : np.ndarray
            Decay Associated Spectra
        """
        return self.estimated_matrix(dataset)

    def decay_associated_spectra(self, dataset: str) -> np.ndarray:
        """

        Parameters
        ----------
        dataset: str
            Label of the dataset.


        Returns
        -------
        das : np.ndarray
            Decay Associated Spectra
        """
        initial_concentration = \
            self.model.datasets[dataset].initial_concentration
        initial_concentration = \
            self.model.initial_concentrations[initial_concentration]
        megacomplex = self.model.datasets[dataset].megacomplexes
        if len(megacomplex) > 1:
            raise NotImplementedError("Retrieving DAS for multiple"
                                      "megacomplexes not supported yet.")
        mat = self.model.get_megacomplex_k_matrix(megacomplex[0])
        return np.dot(
            self.estimated_matrix(dataset),
            mat.a_matrix(initial_concentration, self.best_fit_parameter),
        )

    def concentrations(self, dataset: str) -> np.ndarray:
        """

        Parameters
        ----------
        dataset: str
                Label of the dataset.


        Returns
        -------
        concentrations : np.ndarray
            Concentrations

        """
        return self.calculated_matrix(dataset)
