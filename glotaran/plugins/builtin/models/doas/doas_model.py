""" Gloataran DOAS Model """

from glotaran.model import model
from glotaran.plugins.builtin.models.kinetic_spectrum.kinetic_spectrum_dataset_descriptor import \
    KineticSpectrumDatasetDescriptor
from glotaran.plugins.builtin.models.kinetic_spectrum.kinetic_spectrum_model import (
    KineticSpectrumModel,
    apply_kinetic_model_constraints,
    spectral_constraint_penalty,
    grouped,
    index_dependend,
)

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
    additional_penalty_function=spectral_constraint_penalty,
    grouped=grouped,
    index_dependend=index_dependend,
)
class DOASModel(KineticSpectrumModel):
    """Extends the kinetic model with damped oscillations."""
