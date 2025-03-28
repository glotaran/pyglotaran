from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias
from typing import Union

from pydantic import RootModel
from pydantic import SerializationInfo
from pydantic import model_serializer

from glotaran.model.element import ExtendableElement
from glotaran.model.errors import GlotaranModelError
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.utils.io import serialization_info_to_kwargs

if TYPE_CHECKING:
    from collections.abc import Iterator

ElementType: TypeAlias = Union[tuple(__PluginRegistry.element.values())]  # type:ignore[valid-type] # noqa: UP007
LibraryType: TypeAlias = dict[str, ElementType]


class ModelLibrary(RootModel[LibraryType]):
    root: LibraryType

    def __init__(self, **data: ElementType) -> None:
        super().__init__(**data)

        extended_elements = self._get_extended_elements()

        current_size = len(extended_elements)
        while current_size != 0:
            for label in extended_elements:
                element = self.root[label]
                assert element.extends is not None
                extends = [self.root[label] for label in element.extends]
                if all(e.label not in extended_elements for e in extends):
                    extends += [element]
                    self.root[label] = reduce(lambda a, b: a.extend(b), extends)
                    extended_elements.remove(label)
            if current_size == len(extended_elements):
                msg = "The extended elements could not be resolved because of cyclic dependencies."
                raise GlotaranModelError(msg)
            current_size = len(extended_elements)

    def __iter__(self) -> Iterator[ElementType]:  # type:ignore[override]
        """Create iterator of values."""
        return iter(self.root)

    def __getitem__(self, item_label: str) -> ElementType:
        """Get element for root."""
        return self.root[item_label]

    @classmethod
    def from_dict(cls, spec: dict) -> ModelLibrary:
        return cls(**{label: m | {"label": label} for label, m in spec.items()})

    def _get_extended_elements(self) -> list[str]:
        return [
            label
            for label, element in self.root.items()
            if isinstance(element, ExtendableElement) and element.is_extended()
        ]

    @model_serializer()
    def serialize(self, info: SerializationInfo) -> dict[str, Any]:
        """Serialize ``ModelLibrary`` in a round tippable way.

        Main difference is that the initial copy of ``ExtendableElement`` instances is serialized
        rather than the already extended version.

        Parameters
        ----------
        info : SerializationInfo
            Serialization arguments passed to ``model_dump`` of the top level element.

        Returns
        -------
        dict[str, Any]
        """
        model_dump_kwargs = serialization_info_to_kwargs(info)
        return {
            key: value.model_dump(**model_dump_kwargs)
            if isinstance(value, ExtendableElement) is False
            else value._original.model_dump(**model_dump_kwargs)  # noqa: SLF001
            for key, value in self.root.items()
        }
