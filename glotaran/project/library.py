from __future__ import annotations

from functools import reduce
from typing import TypeAlias
from typing import Union

from pydantic import RootModel

from glotaran.model.element import ExtendableElement
from glotaran.model.errors import GlotaranModelError
from glotaran.plugin_system.base_registry import __PluginRegistry

LibraryType: TypeAlias = dict[  # type:ignore[misc,valid-type]
    str,
    Union[tuple(__PluginRegistry.element.values())],
]


class ModelLibrary(RootModel[LibraryType]):
    root: LibraryType

    def __init__(self, **data):
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
                raise GlotaranModelError(
                    "The extended elements could not be resolved because of cyclic dependencies."
                )
            current_size = len(extended_elements)

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item_label: str):
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
