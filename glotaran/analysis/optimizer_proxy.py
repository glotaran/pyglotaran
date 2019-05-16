import collections
import dask
import dask.distributed as dd
import numpy as np
import xarray as xr
import lmfit

from glotaran.parameter import ParameterGroup

from . import problem_bag, matrix_calculation, residual_calculation
from .result import Result
from .parameter_server import ParameterClient
from .nnls import residual_nnls
from .variable_projection import residual_variable_projection

ResultFuture = collections.namedtuple('ResultFuture', 'clp_label matrix full_clp_label clp residual')


def optimize(scheme, verbose=True, client=None):

    client = client if client else dd.Client(processes=False)
    initial_parameter = scheme.parameter.as_parameter_dict()
    optimization_result_future = client.submit(optimize_task, initial_parameter, scheme, verbose)
    return r rm optimization_result_future.result()


def optimize_task(initial_parameter, scheme, verbose):

    client = dd.get_client()
    with ParameterClient(client) as parameter_client:
        penalty, result = create_problem(scheme, parameter_client)

        minimizer = lmfit.Minimizer(
            calculate_penalty,
            initial_parameter,
            fcn_args=[parameter_client, penalty],
            fcn_kws=None,
            iter_cb=None,
            scale_covar=True,
            nan_policy='omit',
            reduce_fcn=None,
            **{})
        verbose = 2 if verbose else 0
        lm_result = minimizer.minimize(
            method='least_squares', verbose=verbose, max_nfev=scheme.nfev)
        parameter_client.update(lm_result.params)
        parameter = parameter_client.get().result()
        parameter = ParameterGroup.from_parameter_dict(parameter)
        datasets = _create_result_data(parameter, scheme, result)
        covar = lm_result.covar if hasattr(lm_result, 'covar') else None

    return Result(scheme, datasets, parameter,
                  lm_result.nfev, lm_result.nvarys, lm_result.ndata, lm_result.nfree,
                  lm_result.chisqr, lm_result.redchi, lm_result.var_names, covar)


def calculate_penalty(parameter, parameter_client, penalty_job):
    parameter_client.update(parameter).result()
    return penalty_job.compute()


def create_problem(scheme, parameter_client):
    residual_function = residual_nnls if scheme.nnls else residual_variable_projection
    if scheme.model.grouped():
        bag, groups = problem_bag.create_grouped_bag()

        if scheme.model.index_dependend():
            matrix_jobs, combined = \
                matrix_calculation.create_index_dependend_grouped_matrix_jobs(
                    scheme, bag, parameter_client
                )
            clp, residuals, full_residual = \
                residual_calculation.create_index_dependend_grouped_residual(
                    scheme, bag, combined, residual_function
                )
        else:
            matrix_jobs, combined = \
                matrix_calculation.create_index_independend_grouped_matrix_jobs(
                    scheme, groups, parameter_client
                )
            clp, residuals, full_residual = \
                residual_calculation.create_index_independend_grouped_residual(
                    scheme, bag, combined, residual_function
                )
    else:
        bag = problem_bag.create_ungrouped_bag(scheme)

        if scheme.model.index_dependend():
            matrix_jobs, combined = \
                matrix_calculation.create_index_dependend_ungrouped_matrix_jobs(
                    scheme, bag, parameter_client
                )
            clp, residuals, full_residual = \
                residual_calculation.create_index_dependend_ungrouped_residual(
                    scheme, bag, combined, residual_function
                )
        else:
            clp_label, matrix_jobs, constraint_matrix_jobs = \
                matrix_calculation.create_index_independend_ungrouped_matrix_jobs(
                    scheme, parameter_client
                )
            reduced_clp_label, clp, residuals, full_residual = \
                residual_calculation.create_index_independend_ungrouped_residual(
                    scheme, bag, constraint_matrix_jobs, residual_function
                )

            result = ResultFuture(clp_label, matrix_jobs, reduced_clp_label, clp, residuals)

    return full_residual, result


def _create_result_data(parameter, scheme, result):
    clp_labels, matrices, reduced_clp_labels, reduced_clps, residuals = result
    model = scheme.model
    datasets = scheme.data

    for label, dataset in datasets.items():
        if model.grouped():
            clp_label = None
            matrix = []
            for index, problem in self._global_problem.items():
                if isinstance(problem, list):
                    for i, p in enumerate(problem):
                        if p.dataset_descriptor.label == label:
                            matrix.append(matrices[index][i])
                            if clp_label is None:
                                clp_label = clp_labels[index][i]
                else:
                    if problem.dataset_descriptor.label == label:
                        matrix.append(matrices[index])
                        if clp_label is None:
                            clp_label = clp_labels[index]
            dataset.coords['clp_label'] = clp_label
            dataset['matrix'] = ((
                (self._scheme.model.global_dimension),
                (self._scheme.model.matrix_dimension),
                ('clp_label')
            ), matrix)
        else:
            clp_label, matrix, reduced_clp_label, reduced_clp, residual = dask.compute(
                clp_labels[label],
                matrices[label],
                reduced_clp_labels[label],
                reduced_clps[label],
                residuals[label],
            )
            reduced_clp = np.asarray(reduced_clp)
            dataset.coords['clp_label'] = clp_label
            dataset['matrix'] = (((model.matrix_dimension), ('clp_label')), matrix)

            dim1 = dataset.coords[model.global_dimension].size
            dim2 = dataset.coords['clp_label'].size
            dataset['clp'] = ((model.global_dimension, 'clp_label'),
                              np.zeros((dim1, dim2), dtype=np.float64))

            for i, clp in enumerate(reduced_clp_label):
                dataset.clp.loc[{'clp_label': clp}] = reduced_clp[:, i]

            dataset['residual'] = (((model.matrix_dimension), (model.global_dimension)),
                                   np.asarray(residual).T)

    #      residual = []
    #      dim1 = dataset.coords[model.global_dimension].size
    #      dim2 = dataset.coords['clp_label'].size
    #      dataset['clp'] = ((model.global_dimension, 'clp_label'),
    #                        np.zeros((dim1, dim2), dtype=np.float64))
    #      for index, problem in self._global_problem.items():
    #          if isinstance(problem, list):
    #              start = 0
    #              for i, p in enumerate(problem):
    #                  if p.dataset_descriptor.label == label:
    #                      end = start + dataset.coords[self._scheme.model.matrix_dimension].size
    #                      dataset.clp.loc[{self._scheme.model.global_dimension: p.index}] = \
    #                          np.array([full_clp[index][full_clp_label[index].index(i)]
    #                                    if i in full_clp_label[index] else None
    #                                    for i in dataset.coords['clp_label'].values])
    #                      residual.append(residuals[index][start:end])
    #                  else:
    #                      start += self._result_data[p.dataset_descriptor.label]\
    #                              .coords[self._scheme.model.matrix_dimension].size
    #          else:
    #              if problem.dataset_descriptor.label == label:
    #                  dataset.clp.loc[{self._scheme.model.global_dimension: problem.index}] = \
    #                      np.array([full_clp[index][full_clp_label[index].index(i)]
    #                                if i in full_clp_label[index] else None
    #                                for i in dataset.coords['clp_label'].values])
    #                  residual.append(residuals[index])
    #      dataset['residual'] = ((
    #          (self._scheme.model.matrix_dimension),
    #          (self._scheme.model.global_dimension),
    #      ), np.asarray(residual).T)
    #
        if 'weight' in dataset:
            dataset['weighted_residual'] = dataset.residual
            dataset.residual = np.multiply(dataset.weighted_residual, dataset.weight**-1)

        size = dataset.residual.shape[0] * dataset.residual.shape[1]
        dataset.attrs['root_mean_square_error'] = \
            np.sqrt((dataset.residual**2).sum()/size).values

        l, v, r = np.linalg.svd(dataset.residual)

        dataset['residual_left_singular_vectors'] = \
            ((model.matrix_dimension, 'left_singular_value_index'), l)

        dataset['residual_right_singular_vectors'] = \
            (('right_singular_value_index', model.global_dimension), r)

        dataset['residual_singular_values'] = \
            (('singular_value_index'), v)

        # reconstruct fitted data

        dataset['fitted_data'] = dataset.data - dataset.residual

    if callable(model.finalize_data):
        global_clp = {index: xr.DataArray(clp, coords=[('clp_label', full_clp_label[index])])
                      for index, clp in full_clp.items()}
        model.finalize_data(global_clp, parameter, datasets)

    return datasets
