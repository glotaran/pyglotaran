"""The glotaran parameter history package."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from glotaran.parameter.parameters import Parameters

if TYPE_CHECKING:
    from os import PathLike


class ParameterHistory:
    """A class representing a history of parameters."""

    def __init__(self):  # noqa: D107
        self._parameter_labels: list[str] = []
        self._parameters: list[np.ndarray] = []
        self.source_path = "parameter_history.csv"

    @classmethod
    def from_dataframe(cls, history_df: pd.DataFrame) -> ParameterHistory:
        """Create a history from a pandas data frame.

        Parameters
        ----------
        history_df : pd.DataFrame
            The source data frame.

        Returns
        -------
        ParameterHistory
            The created history.
        """
        history = cls()

        history._parameter_labels = history_df.columns

        for parameter_values in history_df.values:
            history._parameters.append(parameter_values)

        return history

    @classmethod
    def from_csv(cls, path: str) -> ParameterHistory:
        """Create a history from a csv file.

        Parameters
        ----------
        path : str
            The path to the csv file.

        Returns
        -------
        ParameterHistory
            The created history.
        """
        df = pd.read_csv(path)
        return cls.from_dataframe(df)

    loader = from_csv

    @property
    def parameter_labels(self) -> list[str]:
        """Return the labels of the parameters in the history.

        Returns
        -------
        list[str]
            A list of parameter labels.
        """
        return self._parameter_labels

    @property
    def parameters(self) -> list[np.ndarray]:
        """Return the parameters in the history.

        Returns
        -------
        list[np.ndarray]
            A list of parameters in the history.
        """
        return self._parameters

    def __len__(self) -> int:
        """Return the number of records in the history."""
        return self.number_of_records

    @property
    def number_of_records(self) -> int:
        """Return the number of records in the history.

        Returns
        -------
        int
            The number of records.
        """
        return len(self._parameters)

    def to_dataframe(self) -> pd.DataFrame:
        """Create a data frame from the history.

        Returns
        -------
        pd.DataFrame
            The created data frame.
        """
        return pd.DataFrame(self._parameters, columns=self.parameter_labels)

    def to_csv(self, file_name: str | PathLike[str], delimiter: str = ","):
        """Write a :class:`ParameterHistory` to a CSV file.

        Parameters
        ----------
        file_name : str
            The path to the CSV file.
        delimiter : str
            The delimiter of the CSV file.
        """
        self.source_path = Path(file_name).as_posix()
        self.to_dataframe().to_csv(file_name, sep=delimiter, index=False)

    def append(self, parameters: Parameters, current_iteration: int = 0):
        """Append :class:`Parameters` to the history.

        Parameters
        ----------
        parameters : Parameters
            The group to append.
        current_iteration: int
            Current iteration of the optimizer.

        Raises
        ------
        ValueError
            Raised if the parameter labels differs from previous.
        """
        (
            parameter_labels,
            parameter_values,
            _,
            _,
        ) = parameters.get_label_value_and_bounds_arrays()
        parameter_labels = ["iteration", *parameter_labels]
        if len(self._parameter_labels) == 0:
            self._parameter_labels = parameter_labels
        if parameter_labels != self.parameter_labels:
            raise ValueError("Cannot append parameters. Parameter labels do not match existing.")

        self._parameters.append(np.array([current_iteration, *parameter_values]))

    def get_parameters(self, index: int) -> np.ndarray:
        """Get parameters for a history index.

        Parameters
        ----------
        index : int
            The history index.

        Returns
        -------
        np.ndarray
            The parameter values at the history index as array.
        """
        return self._parameters[index]
