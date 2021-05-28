from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.builtin.models.kinetic_image.kinetic_baseline_megacomplex import (
    KineticBaselineMegacomplex,
)
from glotaran.builtin.models.kinetic_image.kinetic_decay_megacomplex import KineticDecayMegacomplex
from glotaran.builtin.models.kinetic_image.kinetic_image_model import KineticImageModel
from glotaran.builtin.models.kinetic_spectrum.coherent_artifact_megacomplex import (
    CoherentArtifactMegacomplex,
)
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_dataset_descriptor import (
    KineticSpectrumDatasetDescriptor,
)
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_result import (
    finalize_kinetic_spectrum_result,
)
from glotaran.builtin.models.kinetic_spectrum.spectral_constraints import SpectralConstraint
from glotaran.builtin.models.kinetic_spectrum.spectral_constraints import (
    apply_spectral_constraints,
)
from glotaran.builtin.models.kinetic_spectrum.spectral_irf import IrfSpectralMultiGaussian
from glotaran.builtin.models.kinetic_spectrum.spectral_matrix import spectral_matrix
from glotaran.builtin.models.kinetic_spectrum.spectral_penalties import EqualAreaPenalty
from glotaran.builtin.models.kinetic_spectrum.spectral_penalties import apply_spectral_penalties
from glotaran.builtin.models.kinetic_spectrum.spectral_penalties import has_spectral_penalties
from glotaran.builtin.models.kinetic_spectrum.spectral_relations import SpectralRelation
from glotaran.builtin.models.kinetic_spectrum.spectral_relations import apply_spectral_relations
from glotaran.builtin.models.kinetic_spectrum.spectral_relations import retrieve_related_clps
from glotaran.builtin.models.kinetic_spectrum.spectral_shape import SpectralShape
from glotaran.model import model

if TYPE_CHECKING:
    from glotaran.parameter import ParameterGroup


def has_kinetic_model_constraints(model: KineticSpectrumModel) -> bool:
    return len(model.spectral_relations) + len(model.spectral_constraints) != 0


def apply_kinetic_model_constraints(
    model: KineticSpectrumModel,
    dataset: str,
    parameters: ParameterGroup,
    clp_labels: list[str],
    matrix: np.ndarray,
    index: float,
) -> tuple[list[str], np.ndarray]:
    clp_labels, matrix = apply_spectral_relations(
        model, dataset, parameters, clp_labels, matrix, index
    )
    clp_labels, matrix = apply_spectral_constraints(model, clp_labels, matrix, index)
    return clp_labels, matrix


def retrieve_spectral_clps(
    model: KineticSpectrumModel,
    parameters: ParameterGroup,
    clp_labels: dict[str, list[str] | list[list[str]]],
    reduced_clp_labels: dict[str, list[str] | list[list[str]]],
    reduced_clps: dict[str, list[np.ndarray]],
    data: dict[str, xr.Dataset],
) -> dict[str, list[np.ndarray]]:
    if not has_kinetic_model_constraints(model):
        return reduced_clps

    # Note: we are always in index_dependent case when we have constraints
    clps = {}
    for label in clp_labels:
        clps[label] = []
        for i, index_reduced_clp_labels in enumerate(reduced_clp_labels[label]):
            index_clp_labels = clp_labels[label][i]
            index_reduced_clps = reduced_clps[label][i]
            index_clps = np.zeros((len(index_clp_labels)), dtype=np.float64)
            for j, clp_label in enumerate(index_reduced_clp_labels):
                index_clps[index_clp_labels.index(clp_label)] = index_reduced_clps[j]
            clps[label].append(index_clps)
    clps = retrieve_related_clps(model, parameters, clp_labels, clps, data)
    return clps


def index_dependent(model: KineticSpectrumModel) -> bool:
    return (
        any(
            isinstance(irf, IrfSpectralMultiGaussian) and irf.dispersion_center is not None
            for irf in model.irf.values()
        )
        or len(model.spectral_relations) != 0
        or len(model.spectral_constraints) != 0
        or len(model.weights) != 0
    )


def grouped(model: KineticSpectrumModel):
    return len(model.dataset) != 1


@model(
    "kinetic-spectrum",
    attributes={
        "equal_area_penalties": EqualAreaPenalty,
        "shape": SpectralShape,
        "spectral_constraints": SpectralConstraint,
        "spectral_relations": SpectralRelation,
    },
    dataset_type=KineticSpectrumDatasetDescriptor,
    default_megacomplex_type="kinetic-decay",
    megacomplex_types={
        "coherent-artifact": CoherentArtifactMegacomplex,
        "kinetic-decay": KineticDecayMegacomplex,
        "kinetic-baseline": KineticBaselineMegacomplex,
    },
    model_dimension="time",
    global_matrix=spectral_matrix,
    global_dimension="spectral",
    has_matrix_constraints_function=has_kinetic_model_constraints,
    constrain_matrix_function=apply_kinetic_model_constraints,
    retrieve_clp_function=retrieve_spectral_clps,
    has_additional_penalty_function=has_spectral_penalties,
    additional_penalty_function=apply_spectral_penalties,
    grouped=grouped,
    index_dependent=index_dependent,
    finalize_data_function=finalize_kinetic_spectrum_result,
)
class KineticSpectrumModel(KineticImageModel):
    pass
