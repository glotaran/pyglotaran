"""The model property class."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Mapping
from typing import Sequence
from typing import Union

from glotaran.model.util import wrap_func_as_method
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from glotaran.model.model import Model


ParameterOrLabel = Union[str, Parameter]


def _is_scalar_type(t: type) -> bool:
    if hasattr(t, "__origin__"):
        try:
            return not issubclass(t.__origin__, (Sequence, Mapping))
        except Exception:
            pass
    return True


def _is_sequence_type(t: type) -> bool:
    return not _is_scalar_type(t) and issubclass(t.__origin__, Sequence)


def _is_mapping_type(t: type) -> bool:
    return not _is_scalar_type(t) and issubclass(t.__origin__, Mapping)


def _subtype(t: type) -> type:
    if _is_sequence_type(t):
        return t.__args__[0]
    elif _is_mapping_type(t):
        return t.__args__[1]
    return t


def glotaran_is_parameter_property(self) -> bool:
    return self.glotaran_property_subtype is ParameterOrLabel


def _model_property_getter_factory(cls: type, model_property: ModelProperty):
    @wrap_func_as_method(cls, name=model_property._name)
    def getter(self) -> model_property.glotaran_property_type:
        value = getattr(self, f"_{model_property._name}")
        if value is None:
            value = model_property._default
        return value

    return getter


def _model_property_setter_factory(cls: type, model_property: ModelProperty):
    @wrap_func_as_method(cls, name=model_property._name)
    def setter(self, value: model_property.glotaran_property_type):
        if value is None and not model_property._allow_none:
            raise Exception(
                f"Property '{model_property._name}' of '{cls.__name__}' "
                "is not allowed to set to None."
            )
        if value is not None and model_property.glotaran_is_parameter_property:
            if model_property.glotaran_is_scalar_property and not isinstance(value, Parameter):
                value = Parameter(full_label=str(value))
            elif model_property.glotaran_is_sequence_property and all(
                map(lambda v: not isinstance(v, Parameter), value)
            ):
                value = [Parameter(full_label=str(v)) for v in value]
            elif model_property.glotaran_is_mapping_property and all(
                map(lambda v: not isinstance(v, Parameter), value.values())
            ):
                value = {k: Parameter(full_label=str(v)) for k, v in value.items()}
        setattr(self, f"_{model_property._name}", value)

    return setter


class ModelProperty(property):
    def __init__(
        self, cls: type, name: str, property_type: type, doc: str, default: Any, allow_none: bool
    ):

        self._name = name
        self._allow_none = allow_none
        self._default = default

        if _subtype(property_type) is Parameter:
            if _is_scalar_type(property_type):
                property_type = ParameterOrLabel
            elif _is_sequence_type(property_type):
                property_type = Sequence[ParameterOrLabel]
            elif _is_mapping_type(property_type):
                property_type = Mapping[property_type.__args__[0], ParameterOrLabel]

        self._type = property_type

        super().__init__(
            fget=_model_property_getter_factory(cls, self),
            fset=_model_property_setter_factory(cls, self),
            doc=doc,
        )

    @property
    def glotaran_allow_none(self) -> bool:
        return self._allow_none

    @property
    def glotaran_property_type(self) -> type:
        return self._type

    @property
    def glotaran_is_scalar_property(self) -> bool:
        return _is_scalar_type(self._type)

    @property
    def glotaran_is_sequence_property(self) -> bool:
        return _is_sequence_type(self._type)

    @property
    def glotaran_is_mapping_property(self) -> bool:
        return _is_mapping_type(self._type)

    @property
    def glotaran_property_subtype(self) -> type:
        return _subtype(self._type)

    @property
    def glotaran_is_parameter_property(self) -> bool:
        return self.glotaran_property_subtype is ParameterOrLabel

    def glotaran_replace_parameter_with_labels(self, value: Any) -> dict[str, Any]:
        if not self.glotaran_is_parameter_property or value is None:
            return value
        elif self.glotaran_is_scalar_property:
            return value.full_label
        elif self.glotaran_is_sequence_property:
            return [v.full_label for v in value]
        elif self.glotaran_is_mapping_property:
            return {k: v.full_label for k, v in value.items()}

    def glotaran_validate(
        self, value: Any, model: Model, parameters: ParameterGroup = None
    ) -> list[str]:

        if value is None:
            if self.glotaran_allow_none:
                return []
            else:
                return [f"Property '{self._name}' is none but not allowed to be none."]

        missing_model = []
        if self._name in model.model_items:
            items = getattr(model, self._name)

            if self.glotaran_is_sequence_property:
                for item in value:
                    if item not in items:
                        missing_model.append((self._name, item))
            elif self.glotaran_is_mapping_property:
                for item in value.values():
                    if item not in items:
                        missing_model.append((self._name, item))
            elif value not in items:
                missing_model.append((self._name, value))
        missing_model = [
            f"Missing Model Item: '{name}'['{label}']" for name, label in missing_model
        ]

        missing_parameters = []
        if parameters is not None and self.glotaran_is_parameter_property:
            wanted = value
            if self.glotaran_is_scalar_property:
                wanted = [wanted]
            elif self.glotaran_is_mapping_property:
                wanted = wanted.values()
            for parameter in wanted:
                if not parameters.has(parameter.full_label):
                    missing_parameters.append(parameter.full_label)
        missing_parameters = [f"Missing Parameter: '{p}'" for p in missing_parameters]

        return missing_model + missing_parameters

    def glotaran_fill(self, value: Any, model: Model, parameter: ParameterGroup) -> Any:

        if value is None:
            return None

        if self.glotaran_is_scalar_property:
            if self.glotaran_is_parameter_property:
                value.set_from_group(parameter)
            elif hasattr(model, self._name) and not isinstance(value, bool):
                value = getattr(model, self._name)[value].fill(model, parameter)

        elif self.glotaran_is_sequence_property:
            if self.glotaran_is_parameter_property:
                for v in value:
                    v.set_from_group(parameter)
            elif hasattr(model, self._name):
                value = [getattr(model, self._name)[v].fill(model, parameter) for v in value]

        elif self.glotaran_is_mapping_property:
            if self.glotaran_is_parameter_property:
                for v in value.values():
                    v.set_from_group(parameter)
            elif hasattr(model, self._name):
                value = {
                    k: getattr(model, self._name)[v].fill(model, parameter)
                    for (k, v) in value.items()
                }

        return value

    def glotaran_value_as_markdown(
        self,
        value: Any,
        all_parameters: ParameterGroup | None = None,
        initial_parameters: ParameterGroup | None = None,
    ) -> MarkdownStr:
        md = ""
        if self.glotaran_is_scalar_property:
            md = self.glotaran_format_value(value, all_parameters, initial_parameters)
        elif self.glotaran_is_sequence_property:
            for v in value:
                md += f"\n  * {self.glotaran_format_value(v,all_parameters, initial_parameters)}"
        elif self.glotaran_is_mapping_property:
            for k, v in value.items():
                md += (
                    f"\n  * {k}: "
                    f"{self.glotaran_format_value(v,all_parameters, initial_parameters)}"
                )
        return MarkdownStr(md)

    def glotaran_format_value(
        self,
        value: Any,
        all_parameters: ParameterGroup | None = None,
        initial_parameters: ParameterGroup | None = None,
    ) -> str:
        return (
            value.markdown(all_parameters, initial_parameters)
            if self.glotaran_is_parameter_property
            else str(value)
        )
