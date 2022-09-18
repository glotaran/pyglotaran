"""The DatasetModel class."""

from __future__ import annotations

from glotaran.model_new.item import ModelItem
from glotaran.model_new.item import ModelItemType
from glotaran.model_new.item import ParameterType
from glotaran.model_new.item import item
from glotaran.model_new.megacomplex import Megacomplex


@item
class DatasetModel(ModelItem):
    """A `DatasetModel` describes a dataset in terms of a glotaran model.
    It contains references to model items which describe the physical model for
    a given dataset.

    A general dataset descriptor assigns one or more megacomplexes and a scale
    parameter.
    """

    group: str = "default"
    force_index_dependent: bool = False
    megacomplex: list[ModelItemType[Megacomplex]]
    megacomplex_scale: list[ParameterType] | None = None
    global_megacomplex: list[ModelItemType[Megacomplex]] = None
    global_megacomplex_scale: list[ParameterType] | None = None
    scale: ParameterType | None = None
