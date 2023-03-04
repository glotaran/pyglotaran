"""The result class for global analysis."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from tabulate import tabulate

from glotaran.deprecation import deprecate
from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.io import SavingOptions
from glotaran.io import load_result
from glotaran.io import save_result
from glotaran.model import Model
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.parameter import ParameterHistory
from glotaran.parameter import Parameters
from glotaran.project.dataclass_helpers import exclude_from_dict_field
from glotaran.project.dataclass_helpers import file_loadable_field
from glotaran.project.dataclass_helpers import init_file_loadable_fields
from glotaran.project.scheme import Scheme
from glotaran.utils.io import DatasetMapping
from glotaran.utils.io import create_clp_guide_dataset
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from glotaran.typing import StrOrPath


@dataclass
class Result:
    """The result of a global analysis."""

    number_of_function_evaluations: int
    """The number of function evaluations."""

    success: bool
    """Indicates if the optimization was successful."""

    termination_reason: str
    """The reason (message when) the optimizer terminated"""

    glotaran_version: str
    """The glotaran version used to create the result."""

    free_parameter_labels: list[str]
    """List of labels of the free parameters used in optimization."""

    scheme: Scheme = file_loadable_field(Scheme)  # type:ignore[type-var]

    initial_parameters: Parameters = file_loadable_field(  # type:ignore[type-var]
        Parameters
    )

    optimized_parameters: Parameters = file_loadable_field(  # type:ignore[type-var]
        Parameters
    )

    parameter_history: ParameterHistory = file_loadable_field(  # type:ignore[type-var]
        ParameterHistory
    )
    """The parameter history."""

    optimization_history: OptimizationHistory = file_loadable_field(  # type:ignore[type-var]
        OptimizationHistory
    )
    """The optimization history."""

    data: Mapping[str, xr.Dataset] = file_loadable_field(  # type:ignore[type-var]
        DatasetMapping, is_wrapper_class=True
    )
    """The resulting data as a dictionary of :xarraydoc:`Dataset`.

    Notes
    -----
    The actual content of the data depends on the actual model and can be found in the
    documentation for the model.
    """

    additional_penalty: list[np.ndarray] | None = exclude_from_dict_field(None)
    """A vector with the value for each additional penalty, or None"""

    cost: ArrayLike | None = exclude_from_dict_field(None)
    """The final cost."""

    # The below can be none in case of unsuccessful optimization

    chi_square: float | None = None
    r"""The chi-square of the optimization.

    :math:`\chi^2 = \sum_i^N [{Residual}_i]^2`."""

    covariance_matrix: ArrayLike | None = exclude_from_dict_field(None)
    """Covariance matrix.

    The rows and columns are corresponding to :attr:`free_parameter_labels`."""

    degrees_of_freedom: int | None = None
    """Degrees of freedom in optimization :math:`N - N_{vars} - N_{clps}`."""
    number_of_clps: int | None = None
    """Number of conditionally linear parameters :math:`N_{clps}`."""

    jacobian: ArrayLike | list | None = exclude_from_dict_field(None)
    """Modified Jacobian matrix at the solution

    See also: :func:`scipy.optimize.least_squares`
    """
    number_of_residuals: int | None = None
    """Number of values in the residual vector :math:`N`."""
    number_of_jacobian_evaluations: int | None = None
    """The number of jacobian evaluations."""
    number_of_free_parameters: int | None = None
    """Number of free parameters in optimization :math:`N_{vars}`"""
    optimality: float | None = None
    reduced_chi_square: float | None = None
    r"""The reduced chi-square of the optimization.

    :math:`\chi^2_{red}= {\chi^2} / {(N - N_{vars} - N_{clps})}`.
    """
    root_mean_square_error: float | None = None
    r"""
    The root mean square error the optimization.

    :math:`rms = \sqrt{\chi^2_{red}}`
    """
    source_path: StrOrPath = field(
        default="result.yml", init=False, repr=False, metadata={"exclude_from_dict": True}
    )
    loader: Callable[[StrOrPath], Result] = field(
        default=load_result, init=False, repr=False, metadata={"exclude_from_dict": True}
    )

    def __post_init__(self):
        """Validate fields and cast attributes to correct type."""
        init_file_loadable_fields(self)
        if isinstance(self.jacobian, list):
            self.jacobian = np.array(self.jacobian)
            self.covariance_matrix = np.array(self.covariance_matrix)

    @property
    @deprecate(
        deprecated_qual_name_usage="glotaran.project.Result.number_of_data_points",
        new_qual_name_usage="glotaran.project.Result.number_of_free_parameters",
        to_be_removed_in_version="0.8.0",
        importable_indices=(2, 2),
    )
    def number_of_parameters(self) -> int | None:
        """Return the number of free parameters in optimization :math:`N_{vars}`.

        Returns
        -------
        int | None
            Number of free parameters in optimization :math:`N_{vars}`.
        """
        return self.number_of_free_parameters

    @property
    @deprecate(
        deprecated_qual_name_usage="glotaran.project.Result.number_of_parameters",
        new_qual_name_usage="glotaran.project.Result.number_of_residuals",
        to_be_removed_in_version="0.8.0",
        importable_indices=(2, 2),
    )
    def number_of_data_points(self) -> int | None:
        """Return the number of values in the residual vector :math:`N`.

        Deprecated since it returned the wrong value.

        Returns
        -------
        int | None
            Number of values in the residual vector :math:`N`.
        """
        return self.number_of_residuals

    @property
    def model(self) -> Model:
        """Return the model used to fit result.

        Returns
        -------
        Model
            The model instance.
        """
        return self.scheme.model

    def get_scheme(self) -> Scheme:
        """Return a new scheme from the Result object with optimized parameters.

        Returns
        -------
        Scheme
            A new scheme with the parameters set to the optimized values.
            For the dataset weights the (precomputed) weights from the original scheme are used.
        """
        data = {}

        for label, dataset in self.data.items():
            data[label] = dataset.data.to_dataset(name="data")
            if "weight" in dataset:
                data[label]["weight"] = dataset.weight

        return replace(self.scheme, parameters=self.optimized_parameters)

    def markdown(
        self,
        with_model: bool = True,
        *,
        base_heading_level: int = 1,
        wrap_model_in_details: bool = False,
    ) -> MarkdownStr:
        """Format the model as a markdown text.

        Parameters
        ----------
        with_model : bool
            If `True`, the model will be printed with initial and optimized parameters filled in.
        base_heading_level : int
            The level of the base heading.
        wrap_model_in_details: bool
            Wraps model into details tag. Defaults to ``False``

        Returns
        -------
        MarkdownStr : str
            The scheme as markdown string.
        """
        general_table_rows: list[list[Any]] = [
            ["Number of residual evaluation", self.number_of_function_evaluations],
            ["Number of residuals", self.number_of_residuals],
            ["Number of free parameters", self.number_of_free_parameters],
            ["Number of conditionally linear parameters", self.number_of_clps],
            ["Degrees of freedom", self.degrees_of_freedom],
            ["Chi Square", f"{self.chi_square or np.nan:.2e}"],
            ["Reduced Chi Square", f"{self.reduced_chi_square or np.nan:.2e}"],
            ["Root Mean Square Error (RMSE)", f"{self.root_mean_square_error or np.nan:.2e}"],
        ]
        if self.additional_penalty is not None and any(
            len(penalty) != 0 for penalty in self.additional_penalty
        ):
            general_table_rows.append(["RMSE additional penalty", self.additional_penalty])

        result_table = tabulate(
            general_table_rows,
            headers=["Optimization Result", ""],
            tablefmt="github",
            disable_numparse=True,
        )
        if len(self.data) > 1:
            RMSE_rows = [
                [
                    f"{index}.{label}:",
                    dataset.weighted_root_mean_square_error,
                    dataset.root_mean_square_error,
                ]
                for index, (label, dataset) in enumerate(self.data.items(), start=1)
            ]

            RMSE_table = tabulate(
                RMSE_rows,
                headers=["RMSE (per dataset)", "weighted", "unweighted"],
                floatfmt=".2e",
                tablefmt="github",
            )

            result_table = f"{result_table}\n\n{RMSE_table}"

        if with_model:
            model_md = self.model.markdown(
                parameters=self.optimized_parameters,
                initial_parameters=self.initial_parameters,
                base_heading_level=base_heading_level,
            )
            if wrap_model_in_details is False:
                result_table = f"{result_table}\n\n{model_md}"
            else:
                # The section part is just a hack to generate properly rendering docs due to a bug
                # in sphinx which causes a wrong tag opening and closing order of html tags
                # Since model_md contains 2 heading levels we need to close 2 sections
                result_table = (
                    f"{result_table}\n\n<br><details>\n\n{model_md}\n"
                    f"{'</section>'*(2)}"
                    "</details>"
                    f"{'<section>'*(2)}"
                )

        return MarkdownStr(result_table)

    def _repr_markdown_(self) -> str:
        """Return a markdown representation str.

        Special method used by ``ipython`` to render markdown.

        Returns
        -------
        str
            The scheme as markdown string.
        """
        return str(self.markdown(base_heading_level=3, wrap_model_in_details=True))

    def __str__(self) -> str:
        """Overwrite of ``__str__``."""
        return str(self.markdown(with_model=False))

    def save(
        self, path: StrOrPath, saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT
    ) -> list[str]:
        """Save the result to given folder.

        Parameters
        ----------
        path : StrOrPath
            The path to the folder in which to save the result.
        saving_options : SavingOptions
            Options for the saved result.

        Returns
        -------
        list[str]
            Paths to all the saved files.
        """
        return cast(
            list[str],
            save_result(
                result_path=path,
                result=self,
                format_name="yml",
                allow_overwrite=True,
                saving_options=saving_options,
            ),
        )

    def recreate(self) -> Result:
        """Recrate a result from the initial parameters.

        Returns
        -------
        Result :
            The recreated result.
        """
        from glotaran.optimization.optimize import optimize

        return optimize(self.scheme)

    def verify(self) -> bool:
        """Verify a result.

        Returns
        -------
        bool :
            Weather the recreated result is equal to this result.
        """
        recreated = self.recreate()

        if self.root_mean_square_error != recreated.root_mean_square_error:
            return False

        for label, dataset in self.data.items():
            for attr, array in dataset.items():
                if not np.allclose(array, recreated.data[label][attr]):
                    return False

        return True

    def create_clp_guide_dataset(self, clp_label: str, dataset_name: str) -> xr.Dataset:
        """Create dataset for clp guidance.

        Parameters
        ----------
        clp_label : str
            Label of the clp to guide.
        dataset_name : str
            Name of dataset to extract the guide from.

        Returns
        -------
        xr.Dataset
            DataArray containing the clp guide, with ``clp_label`` dimension replaced by the
            model dimensions first value.

        Raises
        ------
        ValueError
            If ``dataset_name`` is not in result.
        ValueError
            If ``clp_labels`` is not in result.


        Examples
        --------
        Extracting the clp guide from an optimization result object.

        .. code-block:: python

            from glotaran.io import save_dataset

            clp_guide = result.create_clp_guide_dataset("species_1", "dataset_1")
            save_dataset(clp_guide, "clp_guide__result_dataset_1__species_1.nc")


        .. # noqa: DAR402
        """
        return create_clp_guide_dataset(self, clp_label=clp_label, dataset_name=dataset_name)
