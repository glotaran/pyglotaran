"""The package for :class:``Scheme``."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from glotaran.deprecation import deprecate
from glotaran.io import load_dataset
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.project.dataclasses import exclude_from_dict_field
from glotaran.project.dataclasses import file_representation_field
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:

    from typing import Literal

    import xarray as xr

    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup

default_data_filters = {"minimal": ["fitted_data", "residual"], "full": None}


@dataclass
class SavingOptions:
    """A collection of options for result saving."""

    level: Literal["minimal", "full"] = "full"
    data_filter: list[str] | None = None
    data_format: Literal["nc"] = "nc"
    parameter_format: Literal["csv"] = "csv"
    report: bool = True


@dataclass
class Scheme:
    """A scheme is a collection of a model, parameters and a dataset.

    A scheme also holds options for optimization.
    """

    model: Model = exclude_from_dict_field()  # type: ignore
    parameters: ParameterGroup = exclude_from_dict_field()  # type: ignore
    data: dict[str, xr.DataArray | xr.Dataset] = exclude_from_dict_field()  # type: ignore
    model_file: str = file_representation_field("model", load_model, default=None)  # type: ignore # noqa E501
    parameters_file: str = file_representation_field("parameters", load_parameters, None)  # type: ignore # noqa E501
    data_files: dict[str, str] = file_representation_field("data", load_dataset, None)  # type: ignore # noqa E501
    group: bool | None = None
    group_tolerance: float = 0.0
    non_negative_least_squares: bool = False
    maximum_number_function_evaluations: int | None = None
    add_svd: bool = True
    ftol: float = 1e-8
    gtol: float = 1e-8
    xtol: float = 1e-8
    optimization_method: Literal[
        "TrustRegionReflection",
        "Dogbox",
        "Levenberg-Marquardt",
    ] = "TrustRegionReflection"
    saving: SavingOptions = SavingOptions()
    result_path: str | None = None

    def problem_list(self) -> list[str]:
        """Return a list with all problems in the model and missing parameters.

        Returns
        -------
        list[str]
            A list of all problems found in the scheme's model.

        """
        model: Model = self.model
        return model.problem_list(self.parameters)

    def validate(self) -> str:
        """Return a string listing all problems in the model and missing parameters.

        Returns
        -------
        str
            A user-friendly string containing all the problems of a model if any.
            Defaults to 'Your model is valid.' if no problems are found.

        """
        return self.model.validate(self.parameters)

    def markdown(self) -> MarkdownStr:
        """Format the :class:`Scheme` as markdown string.

        Returns
        -------
        MarkdownStr
            The scheme as markdown string.
        """
        model_markdown_str = self.model.markdown(parameters=self.parameters)

        markdown_str = "\n\n"
        markdown_str += "__Scheme__\n\n"

        markdown_str += f"* *nnls*: {self.non_negative_least_squares}\n"
        markdown_str += f"* *nfev*: {self.maximum_number_function_evaluations}\n"
        markdown_str += f"* *group_tolerance*: {self.group_tolerance}\n"

        return model_markdown_str + MarkdownStr(markdown_str)

    def is_grouped(self) -> bool:
        """Return whether the scheme should be grouped.

        Returns
        -------
        bool
            Weather the scheme should be grouped.
        """
        if self.group is not None and not self.group:
            return False
        is_groupable = self.model.is_groupable(self.parameters, self.data)
        if not is_groupable and self.group is not None:
            warnings.warn("Cannot group scheme. Continuing ungrouped.")
        return is_groupable

    def _repr_markdown_(self) -> str:
        """Return a markdown representation str.

        Special method used by ``ipython`` to render markdown.

        Returns
        -------
        str
            The scheme as markdown string.
        """
        return str(self.markdown())

    def __str__(self):
        """Representation used by print and str."""
        return str(self.markdown())

    @staticmethod
    @deprecate(
        deprecated_qual_name_usage="glotaran.project.scheme.Scheme.from_yaml_file(filename)",
        new_qual_name_usage="glotaran.io.load_scheme(filename)",
        to_be_removed_in_version="0.6.0",
        importable_indices=(2, 1),
    )
    def from_yaml_file(filename: str) -> Scheme:
        """Create :class:`Scheme` from specs in yaml file.

        Warning
        -------
        Deprecated use ``glotaran.io.load_scheme(filename)`` instead.

        Parameters
        ----------
        filename : str
            Path to the spec file.

        Returns
        -------
        Scheme
            Analysis schmeme
        """
        return load_scheme(filename)
