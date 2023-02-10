from pydantic import BaseModel

from glotaran.model import Model


class ModelLibrary(BaseModel):
    __root__: dict[str, Model.get_annotated_type()]
