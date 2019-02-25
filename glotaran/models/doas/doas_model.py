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
    matrix=calculate_doas_matrix,
    matrix_dimension='time',
    global_matrix=calculate_doas_spectral_matrix,
    global_dimension='spectral',
    finalize_result_function=finalize_doas_result,
    constrain_matrix_function=apply_kinetic_model_constraints,
    additional_penalty_function=spectral_constraint_penalty,
)
class DOASModel(KineticModel):
    """Extends the kinetic model with damped oscillations."""
