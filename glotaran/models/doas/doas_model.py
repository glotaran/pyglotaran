""" Gloataran DOAS Model """

from glotaran.model import model
from glotaran.models.spectral_temporal import KineticModel, SpectralTemporalDatasetDescriptor
from glotaran.models.spectral_temporal.kinetic_model import (
    apply_kinetic_model_constraints,
    spectral_constraint_penalty
)

from .doas_result import finalize_doas_result
from .doas_megacomplex import DOASMegacomplex
from .doas_matrix import calculate_doas_matrix
from .doas_spectral_matrix import calculate_doas_spectral_matrix
from .oscillation import Oscillation


@model(
    'doas',
    attributes={
        'oscillation': Oscillation,
    },
    dataset_type=SpectralTemporalDatasetDescriptor,
    megacomplex_type=DOASMegacomplex,
    calculated_matrix=calculate_doas_matrix,
    calculated_axis='time',
    estimated_matrix=calculate_doas_spectral_matrix,
    estimated_axis='spectral',
    finalize_result_function=finalize_doas_result,
    constrain_calculated_matrix_function=apply_kinetic_model_constraints,
    additional_penalty_function=spectral_constraint_penalty,
)
class DOASModel(KineticModel):
    """Extends the kinetic model with damped oscillations."""
