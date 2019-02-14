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
    matrix=calculate_kinetic_matrix,
    matrix_dimension='time',
    global_dimension='pixel',
    allow_grouping=False,
)
class FLIMModel(KineticModel):
    """Extends the kinetic model with damped oscillations."""
