from functools import reduce

from pydantic import BaseModel

from glotaran.model import Element
from glotaran.model import ExtendableElement
from glotaran.model.errors import GlotaranModelError


class ModelLibrary(BaseModel):
    __root__: dict[str, Element.get_annotated_type()]

    def __post_init__(self):
        extended_elements = self._get_extended_elements()

        current_size = len(extended_elements)
        while current_size != 0:
            for label, element in extended_elements.items():
                assert element.extends is not None
                extends = [self.__root__[label] for label in element.extends]
                if all(not e.is_extended() for e in extends):
                    extends += [element]
                    self.__root__[label] = reduce(lambda a, b: a.extend(b), extends)
                    del extended_elements[label]
            if current_size == len(extended_elements):
                raise GlotaranModelError(
                    "The extended elements could not be resolved because of cyclic dependencies."
                )
            current_size = len(extended_elements)

    def _get_extended_elements(self) -> dict[str, ExtendableElement]:
        return {
            label: element
            for label, element in self.__root__.items()
            if isinstance(element, ExtendableElement) and element.is_extended()
        }
