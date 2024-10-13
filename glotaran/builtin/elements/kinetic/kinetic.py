"""K-Matrix"""

from __future__ import annotations

import itertools
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import eig
from scipy.linalg import solve

from glotaran.model.item import Item
from glotaran.model.item import ParameterType

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class Kinetic(Item):
    """A scheme for kinetic dynamics, e.g. anergy transfers between states."""

    rates: dict[tuple[str, str], ParameterType]

    @classmethod
    def combine(cls, kinetics: list[Kinetic]) -> Kinetic:
        """Creates a combined matrix.

        When combining k-matrices km1 and km2 (km1.combine(km2)),
        entries in km1 will be overwritten by corresponding entries in km2.

        Parameters
        ----------
        k_matrix :
            KMatrix to combine with.

        Returns
        -------
        combined :
            The combined KMatrix.

        """
        return cls(rates=reduce(lambda lhs, rhs: lhs | rhs, [k.rates for k in kinetics]))

    @property
    def compartments(self) -> list[str]:
        """A list of all compartments involved in the kinetic scheme."""
        return list(dict.fromkeys([c for cs in self.rates for c in reversed(cs)]))

    @property
    def array(self) -> ArrayLike:
        """The reduced representation of the KMatrix as numpy array.

        Parameters
        ----------
        compartments :
            The compartment order.
        """

        size = len(self.compartments)
        array = np.zeros((size, size), dtype=np.float64)
        for (to_comp, from_comp), rate in self.rates.items():
            to_idx = self.compartments.index(to_comp)
            fr_idx = self.compartments.index(from_comp)
            array[to_idx, fr_idx] = rate
        return array

    @property
    def full_array(self) -> ArrayLike:
        """The full representation of the KMatrix as numpy array.

        Parameters
        ----------
        compartments :
            The compartment order.
        """
        size = len(self.compartments)
        array = np.zeros((size, size), np.float64)
        for (to_comp, from_comp), rate in self.rates.items():
            to_idx = self.compartments.index(to_comp)
            fr_idx = self.compartments.index(from_comp)

            if to_idx == fr_idx:
                array[to_idx, fr_idx] -= rate
            else:
                array[to_idx, fr_idx] += rate
                array[fr_idx, fr_idx] -= rate
        return array

    def eigen(self) -> tuple[ArrayLike, ArrayLike]:
        """Returns the eigenvalues and eigenvectors of the k matrix.

        Parameters
        ----------
        compartments :
            The compartment order.
        """
        # We take the transpose to be consistent with timp
        # get the eigenvectors and values, we take the left ones to have
        # computation consistent with TIMP
        eigenvalues, eigenvectors = eig(self.full_array.T, left=True, right=False)
        return (eigenvalues.real, eigenvectors.real)

    def calculate(self, concentrations: ArrayLike | None = None) -> ArrayLike:
        """The resulting rates of the matrix.

        By definition, the eigenvalues of the compartmental model are negative and
        the rates are the negatives of the eigenvalues, thus the eigenvalues need to be
        multiplied with ``-1`` to get rates with the correct sign.

        Parameters
        ----------
        compartments: list[str]
            Names of compartment used to order the matrix.
        initial_concentration: np.ndarray
            The initial concentration.
        """
        if concentrations is not None and self.is_sequential(concentrations):
            return -np.diag(self.full_array)
        eigenvalues, _ = self.eigen()
        return -eigenvalues

    def a_matrix(self, concentrations: ArrayLike) -> ArrayLike:
        """The A matrix of the KMatrix.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        return (
            self.a_matrix_sequential()
            if self.is_sequential(concentrations)
            else self.a_matrix_general(concentrations)
        )

    def a_matrix_general(self, concentrations: ArrayLike) -> np.ndarray:
        """The A matrix of the KMatrix for a general model.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        _, eigenvectors = self.eigen()

        gamma = np.diag(solve(eigenvectors, concentrations))

        a_matrix = eigenvectors @ gamma

        return a_matrix.T

    def a_matrix_sequential(self) -> np.ndarray:
        """The A matrix of the KMatrix for a sequential model.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        array = self.full_array.T
        rates = np.diag(array)

        a_matrix = np.zeros(array.shape, dtype=np.float64)
        a_matrix[0, 0] = 1.0
        for i, j in itertools.product(range(rates.size), range(1, rates.size)):
            if i > j:
                continue
            a_matrix[i, j] = np.prod([rates[m] for m in range(j)]) / np.prod(
                [rates[m] - rates[i] for m in range(j + 1) if i != m]
            )

        return a_matrix

    def is_sequential(self, concentrations: ArrayLike) -> bool:
        """Returns true in the KMatrix represents an unibranched model.

        Parameters
        ----------
        initial_concentration :
            The initial concentration.
        """
        if np.sum(concentrations) != 1:
            return False
        array = self.array
        return not any(
            np.nonzero(array[:, i])[0].size != 1 or i != 0 and array[i, i - 1] == 0
            for i in range(array.shape[1])
        )
