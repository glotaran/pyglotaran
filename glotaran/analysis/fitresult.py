"""This package contains the FitResult object"""

import numpy as np
from lmfit.minimizer import Minimizer


from glotaran.model.dataset import Dataset
from glotaran.model.parameter_group import ParameterGroup


from .grouping import create_group, create_data_group
from .variable_projection import clp_variable_projection, residual_variable_projection


class FitResult:
    """The result of a fit."""

    def __init__(self,
                 model,
                 data,
                 initital_parameter,
                 nnls,
                 ):
        """

        Parameters
        ----------
        lm_result: MinimizerResult
        dataset_results: Dict[str, DatasetResult]

        Returns
        -------
        """
        self.model = model
        self.group = create_group(model, data)
        self.data = data
        self.data_group = create_data_group(model, self.group, data)
        self.initital_parameter = initital_parameter
        self.nnls = nnls
        self._lm_result = None
        self._clp = None

    def minimize(self, verbose: int = 2, max_nfev: int = None):
        parameter = self.initital_parameter.as_parameter_dict(only_fit=True)
        minimizer = Minimizer(
            self._residual,
            parameter,
            fcn_args=[],
            fcn_kws=None,
            iter_cb=None,
            scale_covar=True,
            nan_policy='omit',
            reduce_fcn=None,
            **{})

        self._lm_result = minimizer.minimize(method='least_squares',
                                             verbose=verbose,
                                             max_nfev=max_nfev)
        if not self.nnls and self._clp is None:
            self._clp = clp_variable_projection(self.best_fit_parameter,
                                                self.group, self.model,
                                                self.data, self.data_group)

    @property
    def best_fit_parameter(self) -> ParameterGroup:
        """The best fit parameters."""
        return ParameterGroup.from_parameter_dict(self._lm_result.params)

    def get_dataset(self, label: str):
        """get_dataset returns the DatasetResult for the given dataset.

        Parameters
        ----------
        label : str
            The label of the dataset.

        Returns
        -------
        dataset_result: DatasetResult
            The result for the dataset.
        """
        filled_dataset = self.model.dataset[label].fill(self.model,
                                                        self.best_fit_parameter)
        dataset = self.data[label]

        calculated_axis = dataset.get_axis(self.model.calculated_axis)
        estimated_axis = dataset.get_axis(self.model.estimated_axis)

        calculated_matrix = \
            [self.model.calculated_matrix(filled_dataset,
                                          self.model.compartment,
                                          index,
                                          calculated_axis)
             for index in estimated_axis]

        clp_labels = calculated_matrix[0][0]

        dim1 = len(calculated_matrix)
        dim2 = len(clp_labels)

        calculated_matrix = [c[1] for c in calculated_matrix]
        estimated_matrix = np.empty((dim1, dim2), dtype=np.float64)

        all_idx = [i for i in self.group]

        i = 0
        for idx in estimated_axis:
            idx = (np.abs(all_idx - idx)).argmin()
            _, labels, clp = self._clp[idx]
            estimated_matrix[i, :] = np.asarray([clp[labels.index(c)] for c in clp_labels])
            i += 1

        dim2 = calculated_matrix[0].shape[1]
        result = np.zeros((dim1, dim2), dtype=np.float64)
        for i in range(dim1):
            result[i, :] = np.dot(estimated_matrix[i, :], calculated_matrix[i])
        dataset = Dataset()
        dataset.set_axis(self.model.calculated_axis, calculated_axis)
        dataset.set_axis(self.model.estimated_axis, estimated_axis)
        dataset.set_data(result)
        return dataset

    def _residual(self, parameter):
        parameter = ParameterGroup.from_parameter_dict(parameter)
        return residual_variable_projection(parameter, self.group, self.model,
                                            self.data, self.data_group)

    def __str__(self):
        string = "# Fitresult\n\n"

        # pylint: disable=invalid-name

        ll = 32
        lr = 13

        string += "Optimization Result".ljust(ll-1)
        string += "|"
        string += "|".rjust(lr)
        string += "\n"
        string += "|".rjust(ll, "-")
        string += "|".rjust(lr, "-")
        string += "\n"

        string += "Number of residual evaluation |".rjust(ll)
        string += f"{self._lm_result.nfev} |".rjust(lr)
        string += "\n"
        string += "Number of variables |".rjust(ll)
        string += f"{self._lm_result.nvarys} |".rjust(lr)
        string += "\n"
        string += "Number of datapoints |".rjust(ll)
        string += f"{self._lm_result.ndata} |".rjust(lr)
        string += "\n"
        string += "Negrees of freedom |".rjust(ll)
        string += f"{self._lm_result.nfree} |".rjust(lr)
        string += "\n"
        string += "Chi Square |".rjust(ll)
        string += f"{self._lm_result.chisqr:.6f} |".rjust(lr)
        string += "\n"
        string += "Reduced Chi Square |".rjust(ll)
        string += f"{self._lm_result.redchi:.6f} |".rjust(lr)
        string += "\n"

        string += "\n"
        string += "## Best Fit Parameter\n\n"
        string += f"{self.best_fit_parameter}"
        string += "\n"

        return string
