import collections

import dask.distributed as dd
import db


Problem = collections.namedtuple('Problem', 'index dataset_descriptor axis')


def optimize(job, parameter, verbose, nfev):
    
        parameter = self._scheme.parameter.as_parameter_dict()
        minimizer = lmfit.Minimizer(
            self._calculate_penalty,
            parameter,
            fcn_args=None,
            fcn_kws=None,
            iter_cb=None,
            scale_covar=True,
            nan_policy='omit',
            reduce_fcn=None,
            **{})
        verbose = 2 if verbose else 0
        lm_result = minimizer.minimize(
            method='least_squares', verbose=verbose, max_nfev=self._scheme.nfev)

        self._optimal_parameter = ParameterGroup.from_parameter_dict(lm_result.params)
        self._calculate_result()

        covar = lm_result.covar if hasattr(lm_result, 'covar') else None

        return Result(self._scheme, self._result_data, self._optimal_parameter,
                      lm_result.nfev, lm_result.nvarys, lm_result.ndata, lm_result.nfree,
                      lm_result.chisqr, lm_result.redchi, lm_result.var_names, covar)

