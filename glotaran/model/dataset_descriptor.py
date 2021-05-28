"""The DatasetDescriptor class."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generator
from typing import List

from glotaran.model.attribute import model_attribute
from glotaran.parameter import Parameter

if TYPE_CHECKING:
    from glotaran.model.megacomplex import Megacomplex


@model_attribute(
    properties={
        "megacomplex": List[str],
        "megacomplex_scale": {"type": List[Parameter], "default": None, "allow_none": True},
        "scale": {"type": Parameter, "default": None, "allow_none": True},
    }
)
class DatasetDescriptor:
    """A `DatasetDescriptor` describes a dataset in terms of a glotaran model.
    It contains references to model items which describe the physical model for
    a given dataset.

    A general dataset describtor assigns one or more megacomplexes and a scale
    parameter.
    """

    def iterate_megacomplexes(self) -> Generator[tuple[Parameter | int, Megacomplex | str]]:
        for i, megacomplex in enumerate(self.megacomplex):
            scale = self.megacomplex_scale[i] if self.megacomplex_scale is not None else None
            yield scale, megacomplex
