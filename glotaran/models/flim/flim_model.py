from glotaran.model import model
from glotaran.models.spectral_temporal import (
    KineticModel,
    KineticMegacomplex,
    SpectralTemporalDatasetDescriptor,
)
from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix


@model(
    'flim',
    dataset_type=SpectralTemporalDatasetDescriptor,
    megacomplex_type=KineticMegacomplex,
    calculated_matrix=calculate_kinetic_matrix,
    calculated_axis='time',
    estimated_axis='pixel',
    allow_grouping=False,
)
class FLIMModel(KineticModel):
    """Extends the kinetic model with damped oscillations."""
