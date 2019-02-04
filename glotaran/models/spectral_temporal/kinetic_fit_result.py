import numpy as np
import xarray as xr

from glotaran.analysis.result import Result

from .irf import IrfGaussian
from .spectral_constraints import OnlyConstraint, ZeroConstraint


def finalize_kinetic_result(model, result: Result):

    for label in result.model.dataset:
        dataset = result.data[label]

        dataset_descriptor = result.model.dataset[label].fill(model, result.best_fit_parameter)

        # get_sas

        dataset.coords['species'] = dataset_descriptor.initial_concentration.compartments
        dataset['species_associated_spectra'] = \
            ((result.model.estimated_axis, 'species',),
             dataset.clp.sel(clp_label=dataset_descriptor.initial_concentration.compartments))

        if dataset_descriptor.baseline:
            dataset['baseline'] = dataset.clp.sel(clp_label=f"{dataset_descriptor.label}_baseline")

        for constraint in model.spectral_constraints:
            if isinstance(constraint, (OnlyConstraint, ZeroConstraint)):
                idx = [index for index in dataset.spectral if constraint.applies(index)]
                if constraint.compartment in dataset.coords['species']:
                    dataset.species_associated_spectra\
                        .loc[{'species': constraint.compartment, 'spectral': idx}] \
                        = np.zeros((len(idx)))
                if constraint.compartment == f"{dataset_descriptor.label}_baseline":
                    dataset.baseline.loc[{'spectral': idx}] = np.zeros((len(idx)))

        for relation in model.spectral_relations:
            if relation.compartment in dataset_descriptor.initial_concentration.compartments:
                relation = relation.fill(model, result.best_fit_parameter)
                idx = [index.values for index in dataset.spectral if relation.applies(index)]
                all_idx = np.asarray(list(result.global_clp.keys()))
                clp = []
                for i in idx:
                    j = np.abs(all_idx - i).argmin()
                    j = all_idx[j]
                    clp.append(
                        result.global_clp[j].sel(clp_label=relation.target).values *
                        relation.parameter
                    )
                sas = xr.DataArray(clp, coords=[('spectral', idx)])

                dataset.species_associated_spectra\
                    .loc[{'species': relation.compartment, 'spectral': idx}] = sas

        dataset['species_concentration'] = (
            (model.estimated_axis, model.calculated_axis, 'species',),
            dataset.concentration.sel(
                clp_label=dataset_descriptor.initial_concentration.compartments).values)

        # get_das
        all_das = []
        all_das_labels = []
        for megacomplex in dataset_descriptor.megacomplex:

            k_matrix = megacomplex.full_k_matrix()
            if k_matrix is None:
                continue

            compartments = dataset_descriptor.initial_concentration.compartments

            compartments = [c for c in compartments if c in k_matrix.involved_compartments()]

            a_matrix = k_matrix.a_matrix(dataset_descriptor.initial_concentration)

            das = dataset.species_associated_spectra.sel(species=compartments)@ a_matrix.T

            all_das_labels.append(megacomplex.label)
            all_das.append(
                xr.DataArray(das, coords=[dataset.coords[model.estimated_axis],
                                          ('compartment', compartments)]))

        if all_das:
            dataset.coords['megacomplex'] = all_das_labels
            dataset['decay_associated_spectra'] = xr.concat(all_das, 'megacomplex')

        # get_coherent artifact
        irf = dataset_descriptor.irf

        if isinstance(irf, IrfGaussian):

                index = irf.dispersion_center if irf.dispersion_center \
                     else dataset.coords['spectral'].min().values
                dataset['irf'] = (('time'), irf.calculate(index, dataset.coords['time']))

                if irf.dispersion_center:
                    for i, dispersion in enumerate(
                            irf.calculate_dispersion(dataset.coords['spectral'].values)):
                        dataset[f'center_dispersion_{i+1}'] = (('spectral', dispersion))

                if irf.coherent_artifact:
                    dataset.coords['coherent_artifact_order'] = \
                            list(range(0, irf.coherent_artifact_order+1))
                    dataset['irf_concentration'] = (
                        (model.estimated_axis, model.calculated_axis, 'coherent_artifact_order'),
                        dataset.concentration.sel(clp_label=irf.clp_labels()).values
                    )
                    dataset['irf_associated_spectra'] = (
                        (model.estimated_axis, 'coherent_artifact_order'),
                        dataset.clp.sel(clp_label=irf.clp_labels()).values
                    )
