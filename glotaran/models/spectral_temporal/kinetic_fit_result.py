import numpy as np
import xarray as xr

from glotaran.analysis.fitresult import FitResult

from .irf import IrfGaussian
from .spectral_constraints import OnlyConstraint, ZeroConstraint


def finalize_kinetic_result(model, result: FitResult):

    for label in result.model.dataset:
        dataset = result.data[label]

        dataset_descriptor = result.model.dataset[label].fill(model, result.best_fit_parameter)

        # get_sas

        compartments = []
        for _, matrix in dataset_descriptor.get_k_matrices():
            if matrix is None:
                continue
            for compartment in matrix.involved_compartments():
                if compartment not in compartments:
                    compartments.append(compartment)
        dataset.coords['species'] = compartments
        dataset['species_associated_spectra'] = ((result.model.estimated_axis, 'species',),
                                                 dataset.clp.sel(clp_label=compartments).values)

        for constraint in model.spectral_constraints:
            if isinstance(constraint, (OnlyConstraint, ZeroConstraint)):
                idx = [index for index in dataset.spectral if constraint.applies(index)]
                dataset.species_associated_spectra\
                    .loc[{'species': constraint.compartment, 'spectral': idx}] \
                    = np.zeros((len(idx)))

        for relation in model.spectral_relations:
            idx = [index for index in dataset.spectral if relation.applies(index)]
            dataset.species_associated_spectra\
                .loc[{'species': constraint.compartment, 'spectral': idx}] \
                = dataset.species_associated_spectra\
                .sel({'species': constraint.target, 'spectral': idx}) * relation.parameter

        dataset['species_concentration'] = (
            (model.estimated_axis, model.calculated_axis, 'species',),
            dataset.concentration.sel(clp_label=compartments).values)
        if dataset_descriptor.baseline is not None:
            dataset['baseline'] = dataset.clp.sel(clp_label=f"{dataset_descriptor.label}_baseline")

        # get_das
        all_das = []
        all_das_labels = []
        for megacomplex in dataset_descriptor.megacomplex:

            k_matrix = megacomplex.get_k_matrix()
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
