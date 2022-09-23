"""The DatasetModel class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.model.item import ItemIssue
from glotaran.model.item import ModelItem
from glotaran.model.item import ModelItemType
from glotaran.model.item import ParameterType
from glotaran.model.item import attribute
from glotaran.model.item import item
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import is_exclusive
from glotaran.model.megacomplex import is_unique

if TYPE_CHECKING:
    from glotaran.model.model import Model
    from glotaran.parameter import ParameterGroup


class ExclusiveMegacomplexIssue(ItemIssue):
    def __init__(self, label: str, megacomplex_type: str, is_global: bool):
        self._label = label
        self._type = megacomplex_type
        self._is_global = is_global

    def to_string(self) -> str:
        return (
            f"Exclusive {'global ' if self._is_global else ''}megacomplex '{self._label}' of "
            f"type '{self._type}' cannot be combined with other megacomplexes."
        )


class UniqueMegacomplexIssue(ItemIssue):
    def __init__(self, label: str, megacomplex_type: str, is_global: bool):
        self._label = label
        self._type = megacomplex_type
        self._is_global = is_global

    def to_string(self):
        return (
            f"Unique {'global ' if self._is_global else ''}megacomplex '{self._label}' of "
            f"type '{self._type}' can only be used once per dataset."
        )


def get_megacomplex_issues(
    value: list[str] | None, model: Model, is_global: bool
) -> list[ItemIssue]:
    issues = []

    if value is not None:
        megacomplexes = [model.megacomplex[label] for label in value]
        for megacomplex in megacomplexes:
            megacomplex_type = megacomplex.__class__
            if is_exclusive(megacomplex_type) and len(megacomplexes) > 1:
                issues.append(
                    ExclusiveMegacomplexIssue(megacomplex.label, megacomplex.type, is_global)
                )
            if (
                is_unique(megacomplex_type)
                and len([m for m in megacomplexes if m.__class__ is megacomplex_type]) > 0
            ):
                issues.append(
                    UniqueMegacomplexIssue(megacomplex.label, megacomplex.type, is_global)
                )
    return issues


def validate_megacomplexes(
    value: list[str], model: Model, parameters: ParameterGroup | None
) -> list[ItemIssue]:
    return get_megacomplex_issues(value, model, False)


def validate_global_megacomplexes(
    value: list[str] | None, model: Model, parameters: ParameterGroup | None
) -> list[ItemIssue]:
    return get_megacomplex_issues(value, model, False)


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
    megacomplex: list[ModelItemType[Megacomplex]] = attribute(validator=validate_megacomplexes)
    megacomplex_scale: list[ParameterType] | None = None
    global_megacomplex: list[ModelItemType[Megacomplex]] | None = attribute(
        alias="megacomplex", default=None, validator=validate_global_megacomplexes
    )
    global_megacomplex_scale: list[ParameterType] | None = None
    scale: ParameterType | None = None
