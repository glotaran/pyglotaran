from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..kinetic_image.irf import IrfMultiGaussian
from ..kinetic_image.kinetic_image_result import retrieve_decay_assocatiated_data
from ..kinetic_image.kinetic_image_result import retrieve_irf
from ..kinetic_image.kinetic_image_result import retrieve_species_assocatiated_data
from .spectral_constraints import OnlyConstraint
from .spectral_constraints import ZeroConstraint
from .spectral_irf import IrfGaussianCoherentArtifact
from .spectral_irf import IrfSpectralMultiGaussian

if TYPE_CHECKING:
    from typing import Dict
    from typing import List
    from typing import Union

    import xarray as xr

    from glotaran.parameter import ParameterGroup

    from .kinetic_spectrum_model import KineticSpectrumModel


def finalize_kinetic_spectrum_result(
    model: KineticSpectrumModel,
    global_indices: List[List[object]],
    reduced_clp_labels: Union[Dict[str, List[str]], np.ndarray],
    reduced_clps: Union[Dict[str, np.ndarray], np.ndarray],
    parameter: ParameterGroup,
    data: Dict[str, xr.Dataset],
):

    for label in model.dataset:
        dataset = data[label]
        dataset_descriptor = model.dataset[label].fill(model, parameter)

        if not dataset_descriptor.get_k_matrices():
            continue

        retrieve_species_assocatiated_data(model, dataset, dataset_descriptor, "spectra")

        if dataset_descriptor.baseline:
            dataset["baseline"] = dataset.clp.sel(clp_label=f"{dataset_descriptor.label}_baseline")

        for constraint in model.spectral_constraints:
            if isinstance(constraint, (OnlyConstraint, ZeroConstraint)):
                idx = [index for index in dataset.spectral if constraint.applies(index)]

        for relation in model.spectral_relations:
            if relation.compartment in dataset.coords["species"]:
                relation = relation.fill(model, parameter)

                # indexes on the global axis
                idx = [index for index in dataset.spectral if relation.applies(index)]
                dataset.species_associated_spectra.loc[
                    {"species": relation.target, model.global_dimension: idx}
                ] = (
                    dataset.species_associated_spectra.sel(
                        {"species": relation.compartment, model.global_dimension: idx}
                    )
                    * relation.parameter
                )

        retrieve_decay_assocatiated_data(model, dataset, dataset_descriptor, "spectra")

        irf = dataset_descriptor.irf
        if isinstance(irf, IrfMultiGaussian):
            if isinstance(irf.center, list):
                dataset["irf_center"] = irf.center[0].value
                dataset["irf_width"] = irf.width[0].value
            else:
                dataset["irf_center"] = irf.center.value
                dataset["irf_width"] = irf.width.value
        if isinstance(irf, IrfSpectralMultiGaussian):
            index = irf.dispersion_center or dataset.coords[model.global_dimension].min().values

            dataset["irf"] = (("time"), irf.calculate(index, dataset.coords["time"]))

            if irf.dispersion_center:
                for i, dispersion in enumerate(
                    irf.calculate_dispersion(dataset.coords["spectral"].values)
                ):
                    dataset[f"center_dispersion_{i+1}"] = (model.global_dimension, dispersion)
        if isinstance(irf, IrfGaussianCoherentArtifact):
            dataset.coords["coherent_artifact_order"] = list(
                range(1, irf.coherent_artifact_order + 1)
            )
            dataset["coherent_artifact_concentration"] = (
                (model.model_dimension, "coherent_artifact_order"),
                dataset.matrix.sel(clp_label=irf.clp_labels()).values,
            )
            dataset["coherent_artifact_associated_spectra"] = (
                (model.global_dimension, "coherent_artifact_order"),
                dataset.clp.sel(clp_label=irf.clp_labels()).values,
            )

        else:
            retrieve_irf(model, dataset, dataset_descriptor, "images")
