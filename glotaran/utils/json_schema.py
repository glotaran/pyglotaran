from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import overload

from pydantic import create_model
from pydantic.json_schema import GenerateJsonSchema

from glotaran.io import load_parameters
from glotaran.model.data_model import DataModel
from glotaran.project import Scheme

if TYPE_CHECKING:
    from pydantic.json_schema import CoreSchemaOrField
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema
    from pydantic_core import core_schema

    from glotaran.parameter import Parameters
    from glotaran.typing.types import StrOrPath

# Core schema types whose JSON schema definition has its own title.
_TITLED_CORE_SCHEMA_TYPES = frozenset({"model", "dataclass", "typed-dict", "enum"})
_FUNCTION_CORE_SCHEMA_TYPES = frozenset({"function-before", "function-after", "function-wrap"})


def _has_own_title(schema: CoreSchema) -> bool:
    """Check if the JSON schema of ``schema`` has its own title, ignoring validator functions."""
    while schema["type"] in _FUNCTION_CORE_SCHEMA_TYPES:
        schema = schema["schema"]
    return schema["type"] in _TITLED_CORE_SCHEMA_TYPES


class _TypeAliasFieldTitleGenerateJsonSchema(GenerateJsonSchema):
    """JSON schema generator that sets the title of fields annotated with a ``type`` alias.

    Pydantic does not set the title of a field whose schema has or references a definition, since
    a model or enum definition has its own title. Fields annotated with a ``type`` alias, such as
    ``ParameterType``, also have or reference a definition, but that definition has no title.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self._core_definitions: dict[str, CoreSchema] = {}

    def definitions_schema(self, schema: core_schema.DefinitionsSchema) -> JsonSchemaValue:
        self._core_definitions |= {
            definition["ref"]: definition for definition in schema["definitions"]
        }
        return super().definitions_schema(schema)

    def field_title_should_be_set(self, schema: CoreSchemaOrField) -> bool:
        definition: CoreSchema | None = None
        if schema["type"] == "definition-ref":
            definition = self._core_definitions.get(schema["schema_ref"])  # type:ignore[typeddict-item]
        elif "ref" in schema:
            # Definitions that are only used once are inlined with their ref
            definition = schema
        if definition is not None and not _has_own_title(definition):
            return True
        return super().field_title_should_be_set(schema)


@lru_cache
def _create_vanilla_schema_cached() -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate schema for :class:`Scheme` and a new ``GlotaranDataModel`` class.

    Since the actual GlotaranDataModel is created dynamically we create a schema that has the
    attributes all subclasses.

    We do this in a cached function to:
    - Save time recreating the schema (usage with file watcher to update parameters)
    - Not recreate ``data_model_class`` since this will lead to issues when run in an interactive
        session since each ``GlotaranDataModel`` class is a subclass of :class:`DataModel`.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Json Schema for :class:`Scheme` and a :class:`DataModel`
    """
    data_model_class = create_model(
        "GlotaranDataModel",
        __base__=tuple(
            subclass
            for subclass in DataModel.__subclasses__()
            if subclass.__qualname__.startswith("GlotaranDataModel_") is False
        ),
    )
    return (
        Scheme.model_json_schema(schema_generator=_TypeAliasFieldTitleGenerateJsonSchema),
        data_model_class.model_json_schema(
            schema_generator=_TypeAliasFieldTitleGenerateJsonSchema
        ),
    )


def _create_vanilla_schema() -> tuple[dict[str, Any], dict[str, Any]]:
    """Extra layer since mutable objects and cached return values don't play nicely.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Copy of cached Json Schema from ``_create_vanilla_schema_cached``.
    """
    json_schema, data_model_schema = _create_vanilla_schema_cached()
    return deepcopy(json_schema), deepcopy(data_model_schema)


@overload
def create_model_scheme_json_schema(
    output_path: None = None, parameters: Parameters | StrOrPath | None = None
) -> dict[str, Any]: ...


@overload
def create_model_scheme_json_schema(
    output_path: StrOrPath, parameters: Parameters | StrOrPath | None = None
) -> None: ...


def create_model_scheme_json_schema(
    output_path: StrOrPath | None = None, parameters: Parameters | StrOrPath | None = None
) -> dict[str, Any] | None:
    """Create the json scheme for the model scheme.

    Parameters
    ----------
    output_path : StrOrPath | None
        Path to write the schema to. Defaults to None
    parameters : Parameters | StrOrPath | None
        Parameters to inject labels into schema. Defaults to None

    Returns
    -------
    dict[str, Any] | None
        Json Schema dict if no ``output_path`` was provided.
    """
    json_schema, data_model_schema = _create_vanilla_schema()

    # Overwrite required fields with required fields from base class
    data_model_schema["required"] = [
        name for name, field in DataModel.model_fields.items() if field.is_required()
    ]
    json_schema["$defs"] |= data_model_schema.pop("$defs")
    json_schema["$defs"]["GlotaranDataModel"] = data_model_schema
    json_schema["$defs"]["ExperimentModel"]["properties"]["datasets"]["additionalProperties"][
        "$ref"
    ] = "#/$defs/GlotaranDataModel"

    # We need this overwrite since we generate json schema for editor support (auto completion and
    # linting) of model schema files where the parameter label is used rather than defining a
    # parameter.
    parameter_label_schema: dict[str, str | list[str]] = {
        "description": "Label of a parameter in the parameters file.",
        "title": "ParameterLabel",
        "type": "string",
    }
    if parameters is not None:
        if isinstance(parameters, str | Path):
            parameters = load_parameters(parameters)
        parameter_label_schema |= {"enum": parameters.labels}
    json_schema["$defs"]["Parameter"] = parameter_label_schema

    if output_path is None:
        return json_schema

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="w", encoding="utf8") as f:
        json.dump(json_schema, f)
    return None
