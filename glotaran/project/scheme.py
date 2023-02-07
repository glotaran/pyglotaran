"""The module for :class:``Scheme``."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from glotaran.io import load_scheme
from glotaran.model import Model
from glotaran.parameter import Parameters
from glotaran.project.dataclass_helpers import file_loadable_field
from glotaran.project.dataclass_helpers import init_file_loadable_fields
from glotaran.utils.io import DatasetMapping
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from typing import Literal

    import xarray as xr

    from glotaran.typing import StrOrPath


@dataclass
class Scheme:
    """A scheme is a collection of a model, parameters and a dataset.

    A scheme also holds options for optimization.
    """

    model: Model = file_loadable_field(Model)  # type:ignore[type-var]
    parameters: Parameters = file_loadable_field(Parameters)  # type:ignore[type-var]
    data: Mapping[str, xr.Dataset] = file_loadable_field(
        DatasetMapping, is_wrapper_class=True
    )  # type:ignore[type-var]

    clp_link_tolerance: float = 0.0
    clp_link_method: Literal["nearest", "backward", "forward"] = "nearest"

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
    source_path: StrOrPath = field(
        default="scheme.yml", init=False, repr=False, metadata={"exclude_from_dict": True}
    )
    loader: Callable[[StrOrPath], Scheme] = field(
        default=load_scheme, init=False, repr=False, metadata={"exclude_from_dict": True}
    )

    def __post_init__(self):
        """Override attributes after initialization."""
        init_file_loadable_fields(self)

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
