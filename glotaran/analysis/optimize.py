import collections
import dask
import dask.distributed as dd
import numpy as np
import lmfit

from glotaran.parameter import ParameterGroup

from . import problem_bag, matrix_calculation, residual_calculation
from .result import Result
from .parameter_server import ParameterClient
from .nnls import residual_nnls
from .variable_projection import residual_variable_projection

ResultFuture = \
    collections.namedtuple('ResultFuture', 'bag clp_label matrix full_clp_label clp residual')


def optimize(scheme, verbose=True, client=None):

    if client is None:
        try:
            client = dd.get_client()
        except Exception:
            client = dd.Client(processes=False)
    initial_parameter = scheme.parameter.as_parameter_dict()
    scheme = client.scatter(scheme)
    optimization_result_future = client.submit(optimize_task, initial_parameter, scheme, verbose)
    return optimization_result_future.result()


def optimize_task(initial_parameter, scheme, verbose):

    client = dd.get_client()
    with ParameterClient(client) as parameter_client:
        penalty, result = create_problem(scheme, client, parameter_client)

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
    penalty = penalty_job.compute()
    return penalty


def create_problem(scheme, client, parameter_client):
    residual_function = residual_nnls if scheme.nnls else residual_variable_projection
    if scheme.model.grouped():
        bag, groups = problem_bag.create_grouped_bag(scheme)
        bag = bag.persist()

        if scheme.model.index_dependend():
            clp_label, matrix_jobs, constraint_matrix_jobs = \
                matrix_calculation.create_index_dependend_grouped_matrix_jobs(
                    scheme, bag, parameter_client
                )
            reduced_clp_label, clp, residuals, full_residual = \
                residual_calculation.create_index_dependend_grouped_residual(
                    scheme, bag, constraint_matrix_jobs, residual_function
                )
        else:
            clp_label, matrix_jobs, constraint_matrix_jobs = \
                matrix_calculation.create_index_independend_grouped_matrix_jobs(
                    scheme, groups, parameter_client
                )
            reduced_clp_label, clp, residuals, full_residual = \
                residual_calculation.create_index_independend_grouped_residual(
                    scheme, bag, constraint_matrix_jobs, residual_function
                )
    else:
        bag = problem_bag.create_ungrouped_bag(scheme)

        if scheme.model.index_dependend():
            clp_label, matrix_jobs, constraint_matrix_jobs = \
                matrix_calculation.create_index_dependend_ungrouped_matrix_jobs(
                    scheme, bag, parameter_client
                )
            reduced_clp_label, clp, residuals, full_residual = \
                residual_calculation.create_index_dependend_ungrouped_residual(
                    scheme, bag, constraint_matrix_jobs, residual_function
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

    result = ResultFuture(bag, clp_label, matrix_jobs, reduced_clp_label, clp, residuals)

    return full_residual, result


def _create_result_data(parameter, scheme, result):
    bag, clp_labels, matrices, reduced_clp_labels, reduced_clps, residuals = result
    model = scheme.model
    datasets = scheme.data
    indices = None

    if model.grouped():
        groups = bag.map(lambda group: [d.dataset for d in group.descriptor])
        indices = bag.map(lambda group: [d.index for d in group.descriptor])
        if model.index_dependend():
            groups, indices, clp_labels, matrices, reduced_clp_labels, reduced_clps, residuals = \
                    dask.compute(groups, indices, clp_labels, matrices,
                                 reduced_clp_labels, reduced_clps, residuals)
        else:
            groups, indices, reduced_clp_labels, reduced_clps, residuals = dask.compute(
                groups, indices, reduced_clp_labels, reduced_clps, residuals)

    for label, dataset in datasets.items():
        if model.grouped():
            if model.index_dependend():
                for i, group in enumerate(groups):
                    if label in group:
                        group_index = group.index(label)
                        if 'matrix' not in dataset:
                            # we assume that the labels are the same, this might not be true in
                            # future models
                            dataset.coords['clp_label'] = clp_labels[i][group_index]

                            dim1 = dataset.coords[model.global_dimension].size
                            dim2 = dataset.coords[model.matrix_dimension].size
                            dim3 = dataset.clp_label.size
                            dataset['matrix'] = \
                                (((model.global_dimension),
                                  (model.matrix_dimension),
                                  ('clp_label')),
                                 np.zeros((dim1, dim2, dim3), dtype=np.float64))
                        dataset.matrix.loc[{model.global_dimension: indices[i][group_index]}] = \
                            matrices[i][group_index]
            else:
                clp_label, matrix = dask.compute(
                    clp_labels[label],
                    matrices[label],
                )
                dataset.coords['clp_label'] = clp_label
                dataset['matrix'] = (((model.matrix_dimension), ('clp_label')), matrix)

            dim1 = dataset.coords[model.global_dimension].size
            dim2 = dataset.coords['clp_label'].size
            dataset['clp'] = ((model.global_dimension, 'clp_label'),
                              np.zeros((dim1, dim2), dtype=np.float64))

            dim1 = dataset.coords[model.matrix_dimension].size
            dim2 = dataset.coords[model.global_dimension].size
            dataset['residual'] = ((model.matrix_dimension, model.global_dimension),
                                   np.zeros((dim1, dim2), dtype=np.float64))
            idx = 0
            for i, group in enumerate(groups):
                if label in group:
                    index = indices[i][group.index(label)]
                    for j, clp in enumerate(reduced_clp_labels[i]):
                        if clp in dataset.clp_label:
                            dataset.clp.loc[{'clp_label': clp, model.global_dimension: index}] = \
                                    reduced_clps[i][j]
                    start = 0
                    for dset in group:
                        if dset == label:
                            break
                        start += datasets[dset].coords[model.matrix_dimension].size
                    end = start + dataset.coords[model.matrix_dimension].size
                    dataset.residual.loc[{model.global_dimension: index}] = residuals[i][start:end]

        else:
            clp_label, matrix, reduced_clp_label, reduced_clp, residual = dask.compute(
                clp_labels[label],
                matrices[label],
                reduced_clp_labels[label],
                reduced_clps[label],
                residuals[label],
            )
            reduced_clp = np.asarray(reduced_clp)

            if model.index_dependend():
                # we assume that the labels are the same, this might not be true in future models
                dataset.coords['clp_label'] = clp_label[0]
                dataset['matrix'] = \
                    (((model.global_dimension), (model.matrix_dimension), ('clp_label')), matrix)
            else:
                dataset.coords['clp_label'] = clp_label
                dataset['matrix'] = (((model.matrix_dimension), ('clp_label')), matrix)

            dim1 = dataset.coords[model.global_dimension].size
            dim2 = dataset.coords['clp_label'].size
            dataset['clp'] = ((model.global_dimension, 'clp_label'),
                              np.zeros((dim1, dim2), dtype=np.float64))

            for i, clp in enumerate(reduced_clp_label):
                if model.index_dependend():
                    idx = dataset.coords[model.global_dimension][i]
                    for j, c in enumerate(clp):
                        dataset.clp.loc[{'clp_label': c, model.global_dimension: idx}] = \
                            reduced_clp[i, j]
                else:
                    dataset.clp.loc[{'clp_label': clp}] = reduced_clp[:, i]

            dataset['residual'] = (((model.matrix_dimension), (model.global_dimension)),
                                   np.asarray(residual).T)

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
        model.finalize_data(indices, reduced_clp_labels, reduced_clps, parameter, datasets)

    return datasets
