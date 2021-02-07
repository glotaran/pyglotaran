from __future__ import annotations

import xarray as xr

from glotaran.analysis.problem import Problem

from ..kinetic_image.irf import IrfMultiGaussian
from ..kinetic_image.kinetic_image_result import retrieve_decay_assocatiated_data
from ..kinetic_image.kinetic_image_result import retrieve_irf
from ..kinetic_image.kinetic_image_result import retrieve_species_assocatiated_data
from .spectral_irf import IrfGaussianCoherentArtifact
from .spectral_irf import IrfSpectralMultiGaussian


def finalize_kinetic_spectrum_result(model, problem: Problem, data: dict[str, xr.Dataset]):

    for label, dataset in data.items():

        dataset_descriptor = problem.filled_dataset_descriptors[label]
        if not dataset_descriptor.get_k_matrices():
            continue

        retrieve_species_assocatiated_data(problem.model, dataset, dataset_descriptor, "spectra")

        if dataset_descriptor.baseline:
            dataset["baseline"] = dataset.clp.sel(clp_label=f"{dataset_descriptor.label}_baseline")

        retrieve_decay_assocatiated_data(problem.model, dataset, dataset_descriptor, "spectra")

        irf = dataset_descriptor.irf
        if isinstance(irf, IrfMultiGaussian):
            if isinstance(irf.center, list):
                dataset["irf_center"] = irf.center[0].value
                dataset["irf_width"] = irf.width[0].value
            else:
                dataset["irf_center"] = irf.center.value
                dataset["irf_width"] = irf.width.value
        if isinstance(irf, IrfSpectralMultiGaussian):
            index = (
                irf.dispersion_center
                or dataset.coords[problem.model.global_dimension].min().values
            )

            dataset["irf"] = (("time"), irf.calculate(index, dataset.coords["time"]))

            if irf.dispersion_center:
                for i, dispersion in enumerate(
                    irf.calculate_dispersion(dataset.coords["spectral"].values)
                ):
                    dataset[f"center_dispersion_{i+1}"] = (
                        problem.model.global_dimension,
                        dispersion,
                    )
        if isinstance(irf, IrfGaussianCoherentArtifact):
            dataset.coords["coherent_artifact_order"] = list(
                range(1, irf.coherent_artifact_order + 1)
            )
            dataset["coherent_artifact_concentration"] = (
                (problem.model.model_dimension, "coherent_artifact_order"),
                dataset.matrix.sel(clp_label=irf.clp_labels()).values,
            )
            dataset["coherent_artifact_associated_spectra"] = (
                (problem.model.global_dimension, "coherent_artifact_order"),
                dataset.clp.sel(clp_label=irf.clp_labels()).values,
            )

        else:
            retrieve_irf(problem.model, dataset, dataset_descriptor, "images")
