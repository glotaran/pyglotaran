""" Gloataran DOAS Model """

from glotaran.model import model
from glotaran.models.spectral_temporal import KineticModel, SpectralTemporalDatasetDescriptor

from .doas_megacomplex import DOASMegacomplex
from .doas_matrix import calculate_doas_matrix
#  from .doas_spectral_matrix import calculate_doas_spectral_matrix
from .oscillation import Oscillation


@model(
    'kinetic',
    attributes={
        'oscillation': Oscillation,
    },
    dataset_type=SpectralTemporalDatasetDescriptor,
    megacomplex_type=DOASMegacomplex,
    calculated_matrix=calculate_doas_matrix,
    calculated_axis='time',
    #  estimated_matrix=calculate_doas_spectral_matrix,
    estimated_axis='spectral',
)
class DOASModel(KineticModel):
    """Extends the kinetic model with damped oscillations."""
