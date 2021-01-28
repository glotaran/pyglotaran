"""The result class for global analysis."""

import os
import warnings
from typing import Dict
from typing import List
from typing import Type

import numpy as np
import xarray as xr
from scipy.optimize import OptimizeResult

import glotaran
from glotaran.parameter import ParameterGroup

from .scheme import Scheme


class Result:
    def __init__(
        self,
        scheme: Scheme,
        data: Dict[str, xr.Dataset],
        optimized_parameter: ParameterGroup,
        least_squares_result: OptimizeResult,
        free_parameter_labels: List[str],
    ):
        """The result of a global analysis.

        Parameters
        ----------
        model :
            A subclass of :class:`glotaran.model.Model`
        data :
            A dictionary containing all datasets with their labels as keys.
        initital_parameter : glotaran.parameter.ParameterGroup
            The initital fit parameter,
        nnls :
            (default = False)
            If `True` non-linear least squares optimizing is used instead of variable projection.
        atol :
            (default = 0)
            The tolerance for grouping datasets along the global axis.
        """
        self._scheme = scheme
        self._data = data
        self._optimized_parameter = optimized_parameter
        self._free_parameter_labels = free_parameter_labels
        self._calculate_statistics(least_squares_result)

    def _calculate_statistics(self, least_squares_result: OptimizeResult):
        self._number_function_evaluation = least_squares_result.nfev
        self._number_jacobian_evaluation = least_squares_result.njev
        self._cost = least_squares_result.cost
        self._optimality = least_squares_result.optimality
        self._number_data_points = least_squares_result.fun.size
        self._number_variables = least_squares_result.x.size
        self._degree_of_freedom = self._number_data_points - self._number_variables
        self._chi_square = np.sum(least_squares_result.fun ** 2)
        self._reduced_chi_square = self._chi_square / self._degree_of_freedom
        self._root_mean_square_error = np.sqrt(self._reduced_chi_square)
        self._jacobian = least_squares_result.jac

        try:
            self._covariance_matrix = np.linalg.inv(self._jacobian.T.dot(self._jacobian))
            standard_errors = np.sqrt(np.diagonal(self._covariance_matrix))
            for label, error in zip(self.free_parameter_labels, standard_errors):
                self.optimized_parameter.get(label).standard_error = error
        except np.linalg.LinAlgError:
            warnings.warn(
                "The resulting Jacobian is singular, cannot compute covariance matrix and "
                "standard errors."
            )
            self._covariance_matrix = None

    @property
    def scheme(self) -> Scheme:
        """The scheme for analysis."""
        return self._scheme

    @property
    def model(self) -> Type["glotaran.model.Model"]:
        """The model for analysis."""
        return self._scheme.model

    @property
    def nnls(self) -> bool:
        """If `True` non-linear least squares optimizing is used instead of variable projection."""
        return self._scheme.nnls

    @property
    def data(self) -> Dict[str, xr.Dataset]:
        """The resulting data as a dictionary of :xarraydoc:`Dataset`.

        Notes
        -----
        The actual content of the data depends on the actual model and can be found in the
        documentation for the model.
        """
        return self._data

    @property
    def number_function_evaluation(self) -> int:
        """The number of function evaluations."""
        return self._number_function_evaluation

    @property
    def number_jacobian_evaluation(self) -> int:
        """The number of jacobian evaluations."""
        return self._number_jacobian_evaluation

    @property
    def number_variables(self) -> int:
        """Number of variables in optimization :math:`N_{vars}`"""
        return self._number_variables

    @property
    def number_data_points(self) -> int:
        """Number of data points :math:`N`."""
        return self._number_data_points

    @property
    def degree_of_freedom(self) -> int:
        """Degrees of freedom in optimization :math:`N - N_{vars}`."""
        return self._degree_of_freedom

    @property
    def chi_square(self) -> float:
        r"""The chi-square of the optimization.

        :math:`\chi^2 = \sum_i^N [{Residual}_i]^2`."""
        return self._chi_square

    @property
    def reduced_chi_square(self) -> float:
        r"""The reduced chi-square of the optimization.

        :math:`\chi^2_{red}= {\chi^2} / {(N - N_{vars})}`.
        """
        return self._reduced_chi_square

    @property
    def root_mean_square_error(self) -> float:
        r"""
        The root mean square error the optimization.

        :math:`rms = \sqrt{\chi^2_{red}}`
        """
        return self._root_mean_square_error

    @property
    def free_parameter_labels(self) -> List[str]:
        """List of labels of the free parameter used in optimization."""
        return self._free_parameter_labels

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Covariance matrix.

        The rows and columns are corresponding to :attr:`free_parameter_labels`."""
        return self._covariance_matrix

    @property
    def optimized_parameter(self) -> ParameterGroup:
        """The optimized parameters."""
        return self._optimized_parameter

    @property
    def initial_parameter(self) -> ParameterGroup:
        """The initital fit parameter."""
        return self._scheme.parameter

    def get_dataset(self, dataset_label: str) -> xr.Dataset:
        """Returns the result dataset for the given dataset label.

        Parameters
        ----------
        dataset_label :
            The label of the dataset.
        """
        try:
            return self.data[dataset_label]
        except KeyError:
            raise Exception(f"Unknown dataset '{dataset_label}'")

    def get_scheme(self) -> Scheme:
        data = {}

        for label, dataset in self:
            data[label] = dataset.data.to_dataset(name=data)
            if "weight" in dataset:
                data[label]["weight"] = dataset.weight

        return Scheme(
            model=self.model,
            parameter=self.optimized_parameter,
            data=data,
            group_tolerance=self.group_tolerance,
            nnls=self.scheme.nnls,
            nfev=self.scheme.nfev,
            ftol=self.scheme.ftol,
            gtol=self.scheme.gtol,
            xtol=self.scheme.xtol,
            optimization_method=self.scheme.optimization_method,
        )

    def save(self, path: str) -> List[str]:
        """Saves the result to given folder.

        Returns a list with paths of all saved items.

        The following files are saved:

        * `result.md`: The result with the model formatted as markdown text.
        * `optimized_parameter.csv`: The optimized parameter as csv file.
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

        csv_path = os.path.join(path, "optimized_parameter.csv")
        self.optimized_parameter.to_csv(csv_path)
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
            If `True`, the model will be printed together with the initial and optimized parameter.
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
        string += f"{self.nfev} |".rjust(lr)
        string += "\n"
        string += "Number of variables |".rjust(ll)
        string += f"{self.nvars} |".rjust(lr)
        string += "\n"
        string += "Number of datapoints |".rjust(ll)
        string += f"{self.ndata} |".rjust(lr)
        string += "\n"
        string += "Degrees of freedom |".rjust(ll)
        string += f"{self.nfree} |".rjust(lr)
        string += "\n"
        string += "Chi Square |".rjust(ll)
        string += f"{self.chisqr:.2e} |".rjust(lr)
        string += "\n"
        string += "Reduced Chi Square |".rjust(ll)
        string += f"{self.red_chisqr:.2e} |".rjust(lr)
        string += "\n"
        string += "Root Mean Square Error |".rjust(ll)
        string += f"{self.root_mean_square_error:.2e} |".rjust(lr)
        string += "\n"

        if with_model:
            string += "\n\n" + self.model.markdown(
                parameter=self.optimized_parameter, initial=self.initial_parameter
            )

        return string

    def __str__(self):
        return self.markdown(with_model=False)
