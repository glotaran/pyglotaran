import typing
import numpy as np
import xarray as xr

import glotaran
from glotaran.parameter import ParameterGroup
from glotaran.models.kinetic_image.irf import IrfGaussian
from glotaran.models.kinetic_image.kinetic_image_result import \
        retrieve_species_assocatiated_data, retrieve_decay_assocatiated_data

from .spectral_irf import IrfSpectralGaussian
from .spectral_constraints import OnlyConstraint, ZeroConstraint


def finalize_kinetic_spectrum_result(
    model: 'glotaran.models.spectral_temporal.KineticModel',
    global_indices: typing.List[typing.List[object]],
    reduced_clp_labels: typing.Union[typing.Dict[str, typing.List[str]], np.ndarray],
    reduced_clps: typing.Union[typing.Dict[str, np.ndarray], np.ndarray],
    parameter: ParameterGroup, data: typing.Dict[str, xr.Dataset],
):

    for label in model.dataset:
        dataset = data[label]
        dataset_descriptor = model.dataset[label].fill(model, parameter)

        if not dataset_descriptor.get_k_matrices():
            continue

        retrieve_species_assocatiated_data(model, dataset, dataset_descriptor, "spectra")

        if dataset_descriptor.baseline:
            dataset['baseline'] = dataset.clp.sel(clp_label=f"{dataset_descriptor.label}_baseline")

        for constraint in model.spectral_constraints:
            if isinstance(constraint, (OnlyConstraint, ZeroConstraint)):
                idx = [index for index in dataset.spectral if constraint.applies(index)]
                if constraint.compartment in dataset.coords['species']:
                    dataset.species_associated_spectra\
                        .loc[{'species': constraint.compartment, model.global_dimension: idx}] \
                        = np.zeros((len(idx)))
                if constraint.compartment == f"{dataset_descriptor.label}_baseline":
                    dataset.baseline.loc[{model.global_dimension: idx}] = np.zeros((len(idx)))

        for relation in model.spectral_relations:
            if relation.compartment in dataset_descriptor.initial_concentration.compartments:
                relation = relation.fill(model, parameter)
                idx = [index.values for index in dataset.spectral if relation.applies(index)]
                all_idx = np.asarray(np.sum([idxs for idxs in global_indices]))
                clp = []
                for i in idx:
                    group_idx = np.abs(all_idx - i).argmin()
                    clp_idx = reduced_clp_labels[group_idx].index(relation.target)
                    clp.append(
                        reduced_clps[group_idx][clp_idx] * relation.parameter
                    )
                sas = xr.DataArray(clp, coords=[(model.global_dimension, idx)])

                dataset.species_associated_spectra\
                    .loc[{'species': relation.compartment, model.global_dimension: idx}] = sas

        retrieve_decay_assocatiated_data(model, dataset, dataset_descriptor, "spectra")

        irf = dataset_descriptor.irf
        if isinstance(irf, IrfGaussian):
            if isinstance(irf, IrfSpectralGaussian):
                index = irf.dispersion_center if irf.dispersion_center \
                     else dataset.coords[model.global_dimension].min().values
                dataset['irf'] = (('time'), irf.calculate(index, dataset.coords['time']))

                if irf.dispersion_center:
                    for i, dispersion in enumerate(
                            irf.calculate_dispersion(dataset.coords['spectral'].values)):
                        dataset[f'center_dispersion_{i+1}'] = \
                                ((model.global_dimension, dispersion))

                if irf.coherent_artifact:
                    dataset.coords['coherent_artifact_order'] = \
                            list(range(1, irf.coherent_artifact_order+1))
                    dataset['irf_concentration'] = (
                        (model.global_dimension,
                         model.matrix_dimension,
                         'coherent_artifact_order'),
                        dataset.matrix.sel(clp_label=irf.clp_labels()).values
                    )
                    dataset['irf_associated_spectra'] = (
                        (model.global_dimension, 'coherent_artifact_order'),
                        dataset.clp.sel(clp_label=irf.clp_labels()).values
                    )
            else:
                index = dataset.coords[model.global_dimension][0].values
                dataset['irf'] = (('time'), irf.calculate(index, dataset.coords['time']))
