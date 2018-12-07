from glotaran.model import model
from glotaran.models.spectral_temporal import (
    KineticModel,
    KineticMegacomplex,
    SpectralTemporalDatasetDescriptor,
)
from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix
from glotaran.models.spectral_temporal.spectral_matrix import calculate_spectral_matrix


@model(
    'flim',
    dataset_type=SpectralTemporalDatasetDescriptor,
    megacomplex_type=KineticMegacomplex,
    calculated_matrix=calculate_kinetic_matrix,
    calculated_axis='time',
    estimated_axis='pixel',
    estimated_matrix=calculate_spectral_matrix,
)
class FLIMModel(KineticModel):
    """Extends the kinetic model with damped oscillations."""
