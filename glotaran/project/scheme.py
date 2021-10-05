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


@dataclass
class Scheme:
    """A scheme is a collection of a model, parameters and a dataset.

    A scheme also holds options for optimization.
    """

    model: Model = exclude_from_dict_field()
    parameters: ParameterGroup = exclude_from_dict_field()
    data: dict[str, xr.DataArray | xr.Dataset] = exclude_from_dict_field()
    model_file: str | None = file_representation_field("model", load_model, default=None)
    parameters_file: str | None = file_representation_field("parameters", load_parameters, None)
    data_files: dict[str, str] | None = file_representation_field("data", load_dataset, None)
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

    def valid(self) -> bool:
        """Check if there are no problems with the model or the parameters.

        Returns
        -------
        bool
            Whether the scheme is valid.
        """
        return self.model.valid(self.parameters)

    def markdown(self):
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

    def __str__(self) -> str:
        """Representation used by print and str."""
        return str(self.markdown())

    @property
    def model_dimensions(self) -> dict[str, str]:
        """Return the dataset model's model dimension.

        Returns
        -------
        dict[str, str]
            A dictionary with the dataset labels as key and the model dimension of
            the dataset as value.
        """
        return {
            dataset_name: self.model.dataset[dataset_name]  # type:ignore[attr-defined]
            .fill(self.model, self.parameters)
            .set_data(self.data[dataset_name])
            .get_model_dimension()
            for dataset_name in self.data
        }

    @property
    def global_dimensions(self) -> dict[str, str]:
        """Return the dataset model's global dimension.

        Returns
        -------
        dict[str, str]
            A dictionary with the dataset labels as key and the global dimension of
            the dataset as value.
        """
        return {
            dataset_name: self.model.dataset[dataset_name]  # type:ignore[attr-defined]
            .fill(self.model, self.parameters)
            .set_data(self.data[dataset_name])
            .get_global_dimension()
            for dataset_name in self.data
        }

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
