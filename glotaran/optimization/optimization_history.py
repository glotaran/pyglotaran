"""Module containing the ``OptimizationHistory`` class."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd

from glotaran.utils.regex import RegexPattern

if TYPE_CHECKING:
    from glotaran.typing import StrOrPath


class OptimizationHistory:
    """Wrapped DataFrame to hold information of the optimization and behaves like a ``DataFrame``.

    Ref.:
    https://stackoverflow.com/a/65375904/3990615
    """

    def __init__(self, data=None, source_path: StrOrPath | None = None) -> None:
        """Ensure DataFrame has the correct columns, is numeric and has iteration as index."""
        self._df = (
            pd.DataFrame(
                data,
                columns=["iteration", "nfev", "cost", "cost_reduction", "step_norm", "optimality"],
            )
            .apply(pd.to_numeric)
            .set_index("iteration")
        )
        if source_path is not None:
            self.source_path = Path(source_path).as_posix()
        else:
            self.source_path = "optimization_history.csv"

    def __getattr__(self, attr: str) -> Any:
        """Access class attribute and fallback to DataFrame attribute if not present.

        Parameters
        ----------
        attr: str
            Name of the attribute to access.

        Returns
        -------
        Any
            Attribute of ``OptimizationHistory`` or the DataFrame
        """
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.data, attr)

    def __getitem__(self, column: str) -> pd.Series:
        """Access DataFrame instead of class items.

        Parameters
        ----------
        column: str
            Name of the column to access.

        Returns
        -------
        pd.Series
            Column of the DataFrame.
        """
        return self.data[column]

    @property
    def data(self) -> pd.DataFrame:
        """Underlying ``DataFrame`` which allows for autocomplete with static analyzers.

        Returns
        -------
        pd.DataFrame
            ``DataFrame`` containing ``OptimizationHistory`` data.
        """
        return self._df

    @classmethod
    def from_stdout_str(
        cls: type[OptimizationHistory], optimize_stdout: str
    ) -> OptimizationHistory:
        """Create ``OptimizationHistory`` instance from ``optimize_stdout``.

        Parameters
        ----------
        optimize_stdout: str
            SciPy optimization stdout string, read out via ``TeeContext.read()``.

        Returns
        -------
        OptimizationHistory
            ``OptimizationHistory`` instance created by parsing ``optimize_stdout``.
        """
        return cls(
            [m.groupdict() for m in RegexPattern.optimization_stdout.finditer(optimize_stdout)]
        )

    @classmethod
    def from_csv(cls: type[OptimizationHistory], path: StrOrPath) -> OptimizationHistory:
        """Read ``OptimizationHistory`` from file.

        Parameters
        ----------
        path : StrOrPath
            The path to the csv file.

        Returns
        -------
        OptimizationHistory
            ``OptimizationHistory`` read from file.
        """
        return cls(pd.read_csv(path), source_path=Path(path).as_posix())

    loader = from_csv

    def to_csv(self, path: StrOrPath, delimiter: str = ","):
        """Write a ``OptimizationHistory`` to a CSV file and set ``source_path``.

        Parameters
        ----------
        path : StrOrPath
            The path to the CSV file.
        delimiter : str
            The delimiter of the CSV file.
        """
        self.source_path = Path(path).as_posix()
        self.data.to_csv(path, sep=delimiter)
