from __future__ import annotations

from glotaran.builtin.models.kinetic_image.initial_concentration import InitialConcentration
from glotaran.builtin.models.kinetic_image.irf import Irf
from glotaran.builtin.models.kinetic_image.irf import IrfMultiGaussian
from glotaran.builtin.models.kinetic_image.k_matrix import KMatrix
from glotaran.builtin.models.kinetic_image.kinetic_baseline_megacomplex import (
    KineticBaselineMegacomplex,
)
from glotaran.builtin.models.kinetic_image.kinetic_decay_megacomplex import KineticDecayMegacomplex
from glotaran.builtin.models.kinetic_image.kinetic_image_dataset_descriptor import (
    KineticImageDatasetDescriptor,
)
from glotaran.builtin.models.kinetic_image.kinetic_image_result import (
    finalize_kinetic_image_result,
)
from glotaran.model import Model
from glotaran.model import model


def index_dependent(model: KineticImageModel) -> bool:
    return any(
        isinstance(irf, IrfMultiGaussian) and irf.shift is not None for irf in model.irf.values()
    )


@model(
    "kinetic-image",
    attributes={
        "initial_concentration": InitialConcentration,
        "k_matrix": KMatrix,
        "irf": Irf,
    },
    dataset_type=KineticImageDatasetDescriptor,
    default_megacomplex_type="kinetic-decay",
    megacomplex_types={
        "kinetic-decay": KineticDecayMegacomplex,
        "kinetic-baseline": KineticBaselineMegacomplex,
    },
    model_dimension="time",
    global_dimension="pixel",
    grouped=False,
    index_dependent=index_dependent,
    finalize_data_function=finalize_kinetic_image_result,
)
class KineticImageModel(Model):
    pass
