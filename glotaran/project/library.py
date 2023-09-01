from functools import reduce

from pydantic import BaseModel

from glotaran.model import Element
from glotaran.model import ExtendableElement
from glotaran.model.errors import GlotaranModelError


class ModelLibrary(BaseModel):
    __root__: dict[str, Element.get_annotated_type()]

    def __init__(self, **data):
        super().__init__(**data)

        extended_elements = self._get_extended_elements()

        current_size = len(extended_elements)
        while current_size != 0:
            for label in extended_elements:
                element = self.__root__[label]
                assert element.extends is not None
                extends = [self.__root__[label] for label in element.extends]
                if all(not e.is_extended() for e in extends):
                    extends += [element]
                    self.__root__[label] = reduce(lambda a, b: a.extend(b), extends)
                    extended_elements.remove(label)
            if current_size == len(extended_elements):
                raise GlotaranModelError(
                    "The extended elements could not be resolved because of cyclic dependencies."
                )
            current_size = len(extended_elements)

    def _get_extended_elements(self) -> list[str]:
        return [
            label
            for label, element in self.__root__.items()
            if isinstance(element, ExtendableElement) and element.is_extended()
        ]
