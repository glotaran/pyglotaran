"""The module for :class:``Scheme``."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import TYPE_CHECKING

from glotaran.deprecation import warn_deprecated
from glotaran.io import load_scheme
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project.dataclass_helpers import exclude_from_dict_field
from glotaran.project.dataclass_helpers import file_loadable_field
from glotaran.project.dataclass_helpers import init_file_loadable_fields
from glotaran.utils.io import DatasetMapping
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:

    from typing import Callable
    from typing import Literal
    from typing import Mapping

    import xarray as xr

    from glotaran.typing import StrOrPath


@dataclass
class Scheme:
    """A scheme is a collection of a model, parameters and a dataset.

    A scheme also holds options for optimization.
    """

    model: Model = file_loadable_field(Model)  # type:ignore[type-var]
    parameters: ParameterGroup = file_loadable_field(ParameterGroup)  # type:ignore[type-var]
    data: Mapping[str, xr.Dataset] = file_loadable_field(
        DatasetMapping, is_wrapper_class=True
    )  # type:ignore[type-var]
    clp_link_tolerance: float = 0.0
    maximum_number_function_evaluations: int | None = None
    non_negative_least_squares: bool | None = exclude_from_dict_field(None)
    group_tolerance: float | None = exclude_from_dict_field(None)
    group: bool | None = exclude_from_dict_field(None)
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
    source_path: StrOrPath = field(
        default="scheme.yml", init=False, repr=False, metadata={"exclude_from_dict": True}
    )
    loader: Callable[[StrOrPath], Scheme] = field(
        default=load_scheme, init=False, repr=False, metadata={"exclude_from_dict": True}
    )

    def __post_init__(self):
        """Override attributes after initialization."""
        init_file_loadable_fields(self)

        # Deprecations
        if self.non_negative_least_squares is not None:
            warn_deprecated(
                deprecated_qual_name_usage=(
                    "glotaran.project.Scheme(..., non_negative_least_squares=...)"
                ),
                new_qual_name_usage="<model_file>dataset_groups.default.residual_function",
                to_be_removed_in_version="0.7.0",
                check_qual_names=(True, False),
                stacklevel=4,
            )

            default_group = self.model.dataset_group_models["default"]
            if self.non_negative_least_squares is True:
                default_group.residual_function = "non_negative_least_squares"
            else:
                default_group.residual_function = "variable_projection"
            for field_item in fields(self):
                if field_item.name == "non_negative_least_squares":
                    field_item.metadata = {}

        if self.group is not None:
            warn_deprecated(
                deprecated_qual_name_usage="glotaran.project.Scheme(..., group=...)",
                new_qual_name_usage="<model_file>dataset_groups.default.link_clp",
                to_be_removed_in_version="0.7.0",
                check_qual_names=(True, False),
                stacklevel=4,
            )
            self.model.dataset_group_models["default"].link_clp = self.group
            for field_item in fields(self):
                if field_item.name == "group":
                    field_item.metadata = {}

        if self.group_tolerance is not None:
            warn_deprecated(
                deprecated_qual_name_usage="glotaran.project.Scheme(..., group_tolerance=...)",
                new_qual_name_usage="glotaran.project.Scheme(..., clp_link_tolerance=...)",
                to_be_removed_in_version="0.7.0",
                stacklevel=4,
            )
            self.clp_link_tolerance = self.group_tolerance

    def problem_list(self) -> list[str]:
        """Return a list with all problems in the model and missing parameters.

        Returns
        -------
        list[str]
            A list of all problems found in the scheme's model.
        """
        return self.model.problem_list(self.parameters)

    def validate(self) -> MarkdownStr:
        """Return a string listing all problems in the model and missing parameters.

        Returns
        -------
        MarkdownStr
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

        markdown_str = "\n\n__Scheme__\n\n"
        if self.non_negative_least_squares is not None:
            markdown_str += f"* *non_negative_least_squares*: {self.non_negative_least_squares}\n"
        markdown_str += (
            "* *maximum_number_function_evaluations*: "
            f"{self.maximum_number_function_evaluations}\n"
        )
        markdown_str += f"* *clp_link_tolerance*: {self.clp_link_tolerance}\n"

        return model_markdown_str + MarkdownStr(markdown_str)

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
