"""The result class for global analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from tabulate import tabulate

from glotaran.deprecation import deprecate
from glotaran.io import save_result
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project.scheme import Scheme
from glotaran.utils.ipython import MarkdownStr


@dataclass
class Result:
    """The result of a global analysis"""

    additional_penalty: np.ndarray | None
    """A vector with the value for each additional penalty, or None"""
    cost: ArrayLike
    data: dict[str, xr.Dataset]
    """The resulting data as a dictionary of :xarraydoc:`Dataset`.

    Notes
    -----
    The actual content of the data depends on the actual model and can be found in the
    documentation for the model.
    """
    free_parameter_labels: list[str]
    """List of labels of the free parameters used in optimization."""
    number_of_function_evaluations: int
    """The number of function evaluations."""
    initial_parameters: ParameterGroup
    optimized_parameters: ParameterGroup
    """The optimized parameters, organized in a :class:`ParameterGroup`"""
    scheme: Scheme
    success: bool
    """Indicates if the optimization was successful."""
    termination_reason: str
    """The reason (message when) the optimizer terminated"""

    # The below can be none in case of unsuccessful optimization
    chi_square: float | None = None
    r"""The chi-square of the optimization.

    :math:`\chi^2 = \sum_i^N [{Residual}_i]^2`."""
    covariance_matrix: ArrayLike | None = None
    """Covariance matrix.

    The rows and columns are corresponding to :attr:`free_parameter_labels`."""
    degrees_of_freedom: int | None = None
    """Degrees of freedom in optimization :math:`N - N_{vars}`."""
    jacobian: ArrayLike | None = None
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

    @property
    def model(self) -> Model:
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

        return Scheme(
            model=self.model,
            parameters=self.optimized_parameters,
            data=data,
            group_tolerance=self.scheme.group_tolerance,
            non_negative_least_squares=self.scheme.non_negative_least_squares,
            maximum_number_function_evaluations=self.scheme.maximum_number_function_evaluations,
            ftol=self.scheme.ftol,
            gtol=self.scheme.gtol,
            xtol=self.scheme.xtol,
            optimization_method=self.scheme.optimization_method,
        )

    def markdown(self, with_model: bool = True, base_heading_level: int = 1) -> MarkdownStr:
        """Formats the model as a markdown text.

        Parameters
        ----------
        with_model :
            If `True`, the model will be printed with initial and optimized parameters filled in.
        """

        general_table_rows = [
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
        """Special method used by ``ipython`` to render markdown."""
        return str(self.markdown(base_heading_level=3))

    def __str__(self):
        return str(self.markdown(with_model=False))

    @deprecate(
        deprecated_qual_name_usage="glotaran.project.result.Result.save(result_path)",
        new_qual_name_usage=(
            "glotaran.io.save_result("
            "result=result, result_path=result_path, "
            'format_name="legacy", allow_overwrite=True'
            ")"
        ),
        to_be_removed_in_version="0.6.0",
        importable_indices=(2, 1),
    )
    def save(self, path: str) -> list[str]:
        """Saves the result to given folder.

        Warning
        -------
        Deprecated use ``save_result(result_path=result_path, result=result,
        format_name="legacy", allow_overwrite=True)`` instead.


        Returns a list with paths of all saved items.
        The following files are saved:

        * `result.md`: The result with the model formatted as markdown text.

        * `optimized_parameters.csv`: The optimized parameter as csv file.

        * `{dataset_label}.nc`: The result data for each dataset as NetCDF file.

        Parameters
        ----------
        path :
            The path to the folder in which to save the result.
        """
        save_result(result_path=path, result=self, format_name="legacy", allow_overwrite=True)

    @deprecate(
        deprecated_qual_name_usage="glotaran.project.result.Result.get_dataset(dataset_label)",
        new_qual_name_usage=("glotaran.project.result.Result.data[dataset_label]"),
        to_be_removed_in_version="0.6.0",
        importable_indices=(2, 2),
    )
    def get_dataset(self, dataset_label: str) -> xr.Dataset:
        """Returns the result dataset for the given dataset label.

        Warning
        -------
        Deprecated use ``glotaran.project.result.Result.data[dataset_label]``
        instead.


        Parameters
        ----------
        dataset_label :
            The label of the dataset.
        """
        try:
            return self.data[dataset_label]
        except KeyError:
            raise ValueError(f"Unknown dataset '{dataset_label}'")
