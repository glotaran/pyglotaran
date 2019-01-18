"""Glotaran Kinetic Model"""

import typing
import numpy as np

from glotaran.model import model, BaseModel

from .initial_concentration import InitialConcentration
from .irf import Irf
from .k_matrix import KMatrix
from .kinetic_fit_result import finalize_kinetic_result
from .kinetic_megacomplex import KineticMegacomplex
from .spectral_constraints import (
    SpectralConstraint, apply_spectral_constraints, spectral_constraint_residual)
from .spectral_relations import SpectralRelation, apply_spectral_relations
from .spectral_shape import SpectralShape
from .spectral_temporal_dataset_descriptor import SpectralTemporalDatasetDescriptor
from .kinetic_matrix import calculate_kinetic_matrix
from .spectral_matrix import calculate_spectral_matrix


def apply_kinetic_model_constraints(model: typing.Type['KineticModel'],
                                    clp_labels: typing.List[str],
                                    matrix: np.ndarray,
                                    index: float):
    clp_labels, matrix = apply_spectral_constraints(model, clp_labels, matrix, index)
    clp_labels, matrix = apply_spectral_relations(model, clp_labels, matrix, index)
    return clp_labels, matrix


@model(
    'kinetic',
    attributes={
        'initial_concentration': InitialConcentration,
        'k_matrix': KMatrix,
        'irf': Irf,
        'shape': SpectralShape,
        'spectral_constraints': SpectralConstraint,
        'spectral_relations': SpectralRelation,
    },
    dataset_type=SpectralTemporalDatasetDescriptor,
    megacomplex_type=KineticMegacomplex,
    calculated_matrix=calculate_kinetic_matrix,
    calculated_axis='time',
    estimated_matrix=calculate_spectral_matrix,
    estimated_axis='spectral',
    finalize_result_function=finalize_kinetic_result,
    constrain_calculated_matrix_function=apply_spectral_constraints,
    additional_residual_function=spectral_constraint_residual,
)
class KineticModel(BaseModel):
    """
    A kinetic model is an implementation for model.Model. It is used describe
    time dependend datasets.
    """
