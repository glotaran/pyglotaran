from functools import reduce
from typing import TypeAlias

from pydantic import BaseModel

from glotaran.model import Element
from glotaran.model import ExtendableElement
from glotaran.model.errors import GlotaranModelError

LibraryType: TypeAlias = dict[  # type:ignore[misc,valid-type]
    str, Element.get_annotated_type()  # type:ignore[index]
]


class ModelLibrary(BaseModel):
    __root__: LibraryType

    def __init__(self, **data):
        super().__init__(**data)

        extended_elements = self._get_extended_elements()

        current_size = len(extended_elements)
        while current_size != 0:
            for label in extended_elements:
                element = self.__root__[label]
                assert element.extends is not None
                extends = [self.__root__[label] for label in element.extends]
                if all(e.label not in extended_elements for e in extends):
                    extends += [element]
                    self.__root__[label] = reduce(lambda a, b: a.extend(b), extends)
                    extended_elements.remove(label)
            if current_size == len(extended_elements):
                raise GlotaranModelError(
                    "The extended elements could not be resolved because of cyclic dependencies."
                )
            current_size = len(extended_elements)

    @classmethod
    def from_dict(cls, spec: dict) -> LibraryType:
        return cls.parse_obj({label: m | {"label": label} for label, m in spec.items()}).__root__

    def _get_extended_elements(self) -> list[str]:
        return [
            label
            for label, element in self.__root__.items()
            if isinstance(element, ExtendableElement) and element.is_extended()
        ]
