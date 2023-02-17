from pydantic import BaseModel

from glotaran.model import Element


class ModelLibrary(BaseModel):
    __root__: dict[str, Element.get_annotated_type()]
