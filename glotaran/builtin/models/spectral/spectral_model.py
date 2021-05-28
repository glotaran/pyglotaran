from glotaran.builtin.models.kinetic_image.kinetic_image_dataset_descriptor import (
    KineticImageDatasetDescriptor,
)
from glotaran.builtin.models.spectral.shape import SpectralShape
from glotaran.builtin.models.spectral.spectral_megacomplex import SpectralMegacomplex
from glotaran.builtin.models.spectral.spectral_result import finalize_spectral_result
from glotaran.model import Model
from glotaran.model import model


@model(
    "spectral-model",
    attributes={
        "shape": SpectralShape,
    },
    dataset_type=KineticImageDatasetDescriptor,
    default_megacomplex_type="spectral",
    megacomplex_types={
        "spectral": SpectralMegacomplex,
    },
    model_dimension="spectral",
    global_dimension="time",
    grouped=False,
    index_dependent=False,
    finalize_data_function=finalize_spectral_result,
)
class SpectralModel(Model):
    pass
