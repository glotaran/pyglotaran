"""The result class for global analysis."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from glotaran.model import Model
from glotaran.parameter import ParameterGroup

from .scheme import Scheme


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
    def initial_parameters(self) -> ParameterGroup:
        """The initital parameters."""
        return self.scheme.parameters

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

    def save(self, path: str) -> list[str]:
        """Saves the result to given folder.

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
        if not os.path.exists(path):
            os.makedirs(path)
        elif not os.path.isdir(path):
            raise Exception(f"The path '{path}' is not a directory.")

        paths = []

        md_path = os.path.join(path, "result.md")
        with open(md_path, "w") as f:
            f.write(self.markdown())
        paths.append(md_path)

        csv_path = os.path.join(path, "optimized_parameters.csv")
        self.optimized_parameters.to_csv(csv_path)
        paths.append(csv_path)

        for label, data in self.data.items():
            nc_path = os.path.join(path, f"{label}.nc")
            data.to_netcdf(nc_path, engine="netcdf4")
            paths.append(nc_path)

        return paths

    def markdown(self, with_model=True) -> str:
        """Formats the model as a markdown text.

        Parameters
        ----------
        with_model :
            If `True`, the model will be printed with initial and optimized parameters filled in.
        """

        ll = 32
        lr = 13

        string = "Optimization Result".ljust(ll - 1)
        string += "|"
        string += "|".rjust(lr)
        string += "\n"
        string += "|".rjust(ll, "-")
        string += "|".rjust(lr, "-")
        string += "\n"

        string += "Number of residual evaluation |".rjust(ll)
        string += f"{self.number_of_function_evaluations} |".rjust(lr)
        string += "\n"
        string += "Number of variables |".rjust(ll)
        string += f"{self.number_of_variables} |".rjust(lr)
        string += "\n"
        string += "Number of datapoints |".rjust(ll)
        string += f"{self.number_of_data_points} |".rjust(lr)
        string += "\n"
        string += "Degrees of freedom |".rjust(ll)
        string += f"{self.degrees_of_freedom} |".rjust(lr)
        string += "\n"
        string += "Chi Square |".rjust(ll)
        string += f"{self.chi_square:.2e} |".rjust(lr)
        string += "\n"
        string += "Reduced Chi Square |".rjust(ll)
        string += f"{self.reduced_chi_square:.2e} |".rjust(lr)
        string += "\n"
        string += "Root Mean Square Error (RMSE) |".rjust(ll)
        string += f"{self.root_mean_square_error:.2e} |".rjust(lr)
        string += "\n"
        if self.additional_penalty is not None:
            string += "RMSE additional penalty |".rjust(ll)
            string += f"{sum(self.additional_penalty):.2e} |".rjust(lr)
            string += "\n"
        if len(self.data) > 1:
            string += "RMSE (per dataset) |".rjust(ll)
            string += "weighted |".rjust(lr)
            string += "\n"
            for index, (label, dataset) in enumerate(self.data.items(), start=1):
                string += f"  {index}. {label}: |".rjust(ll)
                string += f"{dataset.weighted_root_mean_square_error:.2e} |".rjust(lr)
                string += "\n"
            string += "RMSE (per dataset) |".rjust(ll)
            string += "unweighted |".rjust(lr)
            string += "\n"
            for index, (label, dataset) in enumerate(self.data.items(), start=1):
                string += f"  {index}. {label}: |".rjust(ll)
                string += f"{dataset.root_mean_square_error:.2e} |".rjust(lr)
                string += "\n"

        if with_model:
            string += "\n\n" + self.model.markdown(
                parameters=self.optimized_parameters, initial_parameters=self.initial_parameters
            )

        return string

    def __str__(self):
        return self.markdown(with_model=False)
