""" Gloataran DOAS Model """

from glotaran.model import model
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_dataset_descriptor import \
    KineticSpectrumDatasetDescriptor
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_model import (
    KineticSpectrumModel,
    apply_kinetic_model_constraints,
    grouped,
    index_dependend,
)
from glotaran.builtin.models.kinetic_spectrum.spectral_penalties import apply_spectral_penalties

from .doas_result import finalize_doas_data
from .doas_megacomplex import DOASMegacomplex
from .doas_matrix import calculate_doas_matrix
from .doas_spectral_matrix import calculate_doas_spectral_matrix
from .oscillation import Oscillation


@model(
    'doas',
    attributes={
        'oscillation': Oscillation,
    },
    dataset_type=KineticSpectrumDatasetDescriptor,
    megacomplex_type=DOASMegacomplex,
    matrix=calculate_doas_matrix,
    matrix_dimension='time',
    global_matrix=calculate_doas_spectral_matrix,
    global_dimension='spectral',
    finalize_data_function=finalize_doas_data,
    constrain_matrix_function=apply_kinetic_model_constraints,
    additional_penalty_function=apply_spectral_penalties,
    grouped=grouped,
    index_dependend=index_dependend,
)
class DOASModel(KineticSpectrumModel):
    """Extends the kinetic model with damped oscillations."""
