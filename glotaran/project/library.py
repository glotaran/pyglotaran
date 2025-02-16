from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING
from typing import TypeAlias
from typing import Union

from pydantic import RootModel

from glotaran.model.element import ExtendableElement
from glotaran.model.errors import GlotaranModelError
from glotaran.plugin_system.base_registry import __PluginRegistry

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
