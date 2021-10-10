"""The result class for global analysis."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Dict
from typing import List
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from tabulate import tabulate

from glotaran.deprecation import deprecate
from glotaran.io import load_dataset
from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.io import save_result
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.parameter import ParameterHistory
from glotaran.project.dataclass_helpers import exclude_from_dict_field
from glotaran.project.dataclass_helpers import file_representation_field
from glotaran.project.scheme import Scheme
from glotaran.utils.ipython import MarkdownStr


class IncompleteResultError(Exception):
    """Exception raised if mandatory arguments to create a result are missing.

    Since some mandatory fields of result can be either created from file or by
    passing a class instance, the file and instance initialization aren't allowed
    to both be None at the same time, but each is allowed to be ``None`` by its own.
    """


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

    scheme: Scheme = cast(Scheme, exclude_from_dict_field(None))
    scheme_file: str | None = file_representation_field("scheme", load_scheme, None)

    initial_parameters: ParameterGroup = cast(ParameterGroup, exclude_from_dict_field(None))
    initial_parameters_file: str | None = file_representation_field(
        "initial_parameters", load_parameters, None
    )

    optimized_parameters: ParameterGroup = cast(ParameterGroup, exclude_from_dict_field(None))
    """The optimized parameters, organized in a :class:`ParameterGroup`"""
    optimized_parameters_file: str | None = file_representation_field(
        "optimized_parameters", load_parameters, None
    )

    parameter_history: ParameterHistory = cast(ParameterHistory, exclude_from_dict_field(None))
    """The parameter history."""
    parameter_history_file: str | None = file_representation_field(
        "parameter_history", ParameterHistory.from_csv, None
    )

    data: dict[str, xr.Dataset] = cast(Dict[str, xr.Dataset], exclude_from_dict_field(None))
    """The resulting data as a dictionary of :xarraydoc:`Dataset`.

    Notes
    -----
    The actual content of the data depends on the actual model and can be found in the
    documentation for the model.
    """
    data_files: dict[str, str] | None = file_representation_field("data", load_dataset, None)

    additional_penalty: np.ndarray | None = exclude_from_dict_field(None)
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
    """Degrees of freedom in optimization :math:`N - N_{vars}`."""

    jacobian: ArrayLike | list | None = exclude_from_dict_field(None)
    """Modified Jacobian matrix at the solution

    See also: :func:`scipy.optimize.least_squares`
    """
    number_of_data_points: int | None = None
    """Number of data points :math:`N`."""
    number_of_jacobian_evaluations: int | None = None
    """The number of jacobian evaluations."""
    number_of_variables: int | None = None
    """Number of variables in optimization :math:`N_{vars}`"""
    optimality: float | None = None
    reduced_chi_square: float | None = None
    r"""The reduced chi-square of the optimization.

    :math:`\chi^2_{red}= {\chi^2} / {(N - N_{vars})}`.
    """
    root_mean_square_error: float | None = None
    r"""
    The root mean square error the optimization.

    :math:`rms = \sqrt{\chi^2_{red}}`
    """

    def __post_init__(self):
        """Validate fields and cast attributes to correct type."""
        self._check_mandatory_fields()
        if isinstance(self.jacobian, list):
            self.jacobian = np.array(self.jacobian)
            self.covariance_matrix = np.array(self.covariance_matrix)

    def _check_mandatory_fields(self):
        """Check that required fields which can be set from file are not ``None``.

        Raises
        ------
        IncompleteResultError
            If any mandatory field and its file representation is ``None``.
        """
        mandatory_fields = [
            ("scheme", ""),
            ("initial_parameters", ""),
            ("optimized_parameters", ""),
            ("parameter_history", ""),
            ("data", "s"),
        ]
        missing_fields = [
            (mandatory_field, file_post_fix)
            for mandatory_field, file_post_fix in mandatory_fields
            if (
                getattr(self, mandatory_field) is None
                and getattr(self, f"{mandatory_field}_file{file_post_fix}") is None
            )
        ]
        if len(missing_fields) != 0:
            error_message = "Result is missing mandatory fields:\n"
            for missing_field, file_post_fix in missing_fields:
                error_message += (
                    f" - Required filed {missing_field!r} is missing!\n"
                    f"   Set either {missing_field!r} or '{missing_field}_file{file_post_fix}'."
                )
            raise IncompleteResultError(error_message)

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

    def markdown(self, with_model: bool = True, base_heading_level: int = 1) -> MarkdownStr:
        """Format the model as a markdown text.

        Parameters
        ----------
        with_model : bool
            If `True`, the model will be printed with initial and optimized parameters filled in.
        base_heading_level : int
            The level of the base heading.

        Returns
        -------
        MarkdownStr : str
            The scheme as markdown string.
        """
        general_table_rows: list[list[Any]] = [
            ["Number of residual evaluation", self.number_of_function_evaluations],
            ["Number of variables", self.number_of_variables],
            ["Number of datapoints", self.number_of_data_points],
            ["Degrees of freedom", self.degrees_of_freedom],
            ["Chi Square", f"{self.chi_square or np.nan:.2e}"],
            ["Reduced Chi Square", f"{self.reduced_chi_square or np.nan:.2e}"],
            ["Root Mean Square Error (RMSE)", f"{self.root_mean_square_error or np.nan:.2e}"],
        ]
        if self.additional_penalty is not None:
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
            result_table = f"{result_table}\n\n{model_md}"

        return MarkdownStr(result_table)

    def _repr_markdown_(self) -> str:
        """Return a markdown representation str.

        Special method used by ``ipython`` to render markdown.

        Returns
        -------
        str
            The scheme as markdown string.
        """
        return str(self.markdown(base_heading_level=3))

    def __str__(self) -> str:
        """Overwrite of ``__str__``."""
        return str(self.markdown(with_model=False))

    def save(self, path: str) -> list[str]:
        """Save the result to given folder.

        Parameters
        ----------
        path : str
            The path to the folder in which to save the result.

        Returns
        -------
        list[str]
            Paths to all the saved files.
        """
        return cast(
            List[str],
            save_result(result_path=path, result=self, format_name="folder", allow_overwrite=True),
        )

    def recreate(self) -> Result:
        """Recrate a result from the initial parameters.

        Returns
        -------
        Result :
            The recreated result.
        """
        from glotaran.analysis.optimize import optimize

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

    @deprecate(
        deprecated_qual_name_usage="glotaran.project.result.Result.get_dataset(dataset_label)",
        new_qual_name_usage=("glotaran.project.result.Result.data[dataset_label]"),
        to_be_removed_in_version="0.6.0",
        importable_indices=(2, 2),
    )
    def get_dataset(self, dataset_label: str) -> xr.Dataset:
        """Return the result dataset for the given dataset label.

        Warning
        -------
        Deprecated use ``glotaran.project.result.Result.data[dataset_label]``
        instead.


        Parameters
        ----------
        dataset_label : str
            The label of the dataset.

        Returns
        -------
        xr.Dataset :
            The dataset.


        .. # noqa: DAR401
        """
        try:
            return self.data[dataset_label]
        except KeyError:
            raise ValueError(f"Unknown dataset '{dataset_label}'")
