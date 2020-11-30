from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from glotaran.model import model

from ..kinetic_image.kinetic_image_megacomplex import KineticImageMegacomplex
from ..kinetic_image.kinetic_image_model import KineticImageModel
from .kinetic_spectrum_dataset_descriptor import KineticSpectrumDatasetDescriptor
from .kinetic_spectrum_matrix import kinetic_spectrum_matrix
from .kinetic_spectrum_result import finalize_kinetic_spectrum_result
from .spectral_constraints import SpectralConstraint
from .spectral_constraints import apply_spectral_constraints
from .spectral_irf import IrfSpectralMultiGaussian
from .spectral_matrix import spectral_matrix
from .spectral_penalties import EqualAreaPenalty
from .spectral_penalties import apply_spectral_penalties
from .spectral_penalties import has_spectral_penalties
from .spectral_relations import SpectralRelation
from .spectral_relations import apply_spectral_relations
from .spectral_relations import retrieve_related_clps
from .spectral_shape import SpectralShape

if TYPE_CHECKING:
    from typing import List
    from typing import Union

    from glotaran.parameter import ParameterGroup


def has_kinetic_model_constraints(model: KineticSpectrumModel) -> bool:
    return len(model.spectral_relations) + len(model.spectral_constraints) != 0


def apply_kinetic_model_constraints(
    model: KineticSpectrumModel,
    parameter: ParameterGroup,
    clp_labels: List[str],
    matrix: np.ndarray,
    index: float,
):
    clp_labels, matrix = apply_spectral_relations(model, parameter, clp_labels, matrix, index)
    clp_labels, matrix = apply_spectral_constraints(model, clp_labels, matrix, index)
    return clp_labels, matrix


def retrieve_spectral_clps(
    model: KineticSpectrumModel,
    parameter: ParameterGroup,
    clp_labels: List[str],
    reduced_clp_labels: List[str],
    reduced_clps: Union[np.ndarray, List[np.ndarray]],
    global_index: float,
):
    if not has_kinetic_model_constraints(model):
        return reduced_clps

    full_clps = np.zeros((len(clp_labels)), dtype=np.float64)
    for i, label in enumerate(reduced_clp_labels):
        full_clps[clp_labels.index(label)] = reduced_clps[i]
    full_clps = retrieve_related_clps(model, parameter, clp_labels, full_clps, global_index)
    return full_clps


def index_dependent(model: KineticSpectrumModel):
    if any(
        isinstance(irf, IrfSpectralMultiGaussian) and irf.dispersion_center is not None
        for irf in model.irf.values()
    ):
        return True
    if len(model.spectral_relations) != 0:
        return True
    return len(model.spectral_constraints) != 0


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
    megacomplex_type=KineticImageMegacomplex,
    matrix=kinetic_spectrum_matrix,
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
