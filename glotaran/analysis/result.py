"""The result class for global analysis."""

import typing
import os

import numpy as np
import xarray as xr

import glotaran  # noqa F01
from glotaran.parameter import ParameterGroup

from .scheme import Scheme


class Result:

    def __init__(self,
                 scheme: Scheme,
                 data: typing.Dict[str, xr.Dataset],
                 parameter: ParameterGroup,
                 nfev: int,
                 nvars: int,
                 ndata: int,
                 nfree: int,
                 chisqr: float,
                 red_chisqr: float,
                 var_names: typing.List[str],
                 covar: np.ndarray,
                 ):
        """The result of a global analysis.

        Parameters
        ----------
        model :
            A subclass of :class:`glotaran.model.Model`
        data :
            A dictonary containing all datasets with their labels as keys.
        initital_parameter : glotaran.parameter.ParameterGroup
            The initital fit parameter,
        nnls :
            (default = False)
            If `True` non-linear least squaes optimizing is used instead of variable projection.
        atol :
            (default = 0)
            The tolerance for grouping datasets along the global axis.
        """
        self._scheme = scheme
        self._data = data
        self._optimized_parameter = parameter
        self._nfev = nfev
        self._nvars = nvars
        self._ndata = ndata,
        self._nfree = nfree
        self._chisqr = chisqr
        self._red_chisqr = red_chisqr
        self._var_names = var_names
        self._covar = covar

    @property
    def scheme(self) -> Scheme:
        """The scheme for analysis."""
        return self._scheme

    @property
    def model(self) -> typing.Type['glotaran.model.Model']:
        """The model for analysis."""
        return self._scheme.model

    @property
    def nnls(self) -> bool:
        """If `True` non-linear least squaes optimizing is used instead of variable
        projection."""
        return self._scheme.nnls

    @property
    def data(self) -> typing.Dict[str, xr.Dataset]:
        """The resulting data as a dictionary of :xarraydoc:`Dataset`.

        Notes
        -----
        The actual content of the data depends on the actual model and can be found in the
        documentation for the model.
        """
        return self._data

    @property
    def nfev(self) -> int:
        """The number of function evaluations."""
        return self._nfev

    @property
    def nvars(self) -> int:
        """Number of variables in optimization :math:`N_{vars}`"""
        return self._nvars

    @property
    def ndata(self) -> int:
        """Number of data points :math:`N`."""
        return self._ndata

    @property
    def nfree(self) -> int:
        """Degrees of freedom in optimization :math:`N - N_{vars}`."""
        return self._nfree

    @property
    def chisqr(self) -> float:
        """The chi-square of the optimization
        :math:`\chi^2 = \sum_i^N [{Residual}_i]^2`.""" # noqa w605
        return self._chisqr

    @property
    def red_chisqr(self) -> float:
        """The reduced chi-square of the optimization
        :math:`\chi^2_{red}= {\chi^2} / {(N - N_{vars})}`.
        """ # noqa w605
        return self._red_chisqr

    @property
    def root_mean_square_error(self) -> float:
        """
        The root mean square error the optimization
        :math:`rms = \sqrt{\chi^2_{red}}`
        """ # noqa w605
        return np.sqrt(self.red_chisqr)

    @property
    def var_names(self) -> typing.List[str]:
        """Ordered list of variable parameter names used in optimization, and
        useful for understanding the values in :attr:`covar`."""
        return [n.replace('_', '.') for n in self._var_names]

    @property
    def covar(self) -> np.ndarray:
        """Covariance matrix from minimization, with rows and columns
        corresponding to :attr:`var_names`."""
        return self._covar

    @property
    def optimized_parameter(self) -> ParameterGroup:
        """The optimized parameters."""
        return self._optimized_parameter

    @property
    def initial_parameter(self) -> ParameterGroup:
        """The initital fit parameter"""
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

    def save(self,  path: str) -> typing.List[str]:
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

        md_path = os.path.join(path, 'result.md')
        with open(md_path, 'w') as f:
            f.write(self.markdown())
        paths.append(md_path)

        csv_path = os.path.join(path, 'optimized_parameter.csv')
        self.optimized_parameter.to_csv(csv_path)
        paths.append(csv_path)

        for label, data in self.data.items():
            nc_path = os.path.join(path, f"{label}.nc")
            data.to_netcdf(nc_path, engine='netcdf4')
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

        string = "Optimization Result".ljust(ll-1)
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
            string += "\n\n" + self.model.markdown(parameter=self.optimized_parameter,
                                                   initial=self.initial_parameter)

        return string

    def __str__(self):
        return self.markdown(with_model=False)
