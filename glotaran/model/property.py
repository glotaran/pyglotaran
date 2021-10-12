"""The model property class."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from glotaran.model.util import wrap_func_as_method
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup

if TYPE_CHECKING:
    from glotaran.model.model import Model


class ModelProperty(property):
    def __init__(self, cls, name, prop_type, doc, default, allow_none):

        self._name = name
        self._allow_none = allow_none
        self._determine_if_parameter(prop_type)

        self._type = prop_type if not self._is_parameter else Union[str, prop_type]

        @wrap_func_as_method(cls, name=name)
        def setter(that_self, value: self._type):
            if value is None and not self._allow_none:
                raise Exception(
                    f"Property '{name}' of '{cls.__name__}' is not allowed to set to None."
                )
            if self._is_parameter and value is not None:
                if self._is_parameter_value and isinstance(value, str):
                    value = Parameter(full_label=str(value))
                elif self._is_parameter_list and all(isinstance(v, str) for v in value):
                    value = [Parameter(full_label=str(v)) for v in value]
                elif self._is_parameter_dict and all(isinstance(v, str) for v in value.values()):
                    for k, v in value.items():
                        value[k] = Parameter(full_label=v)
            setattr(that_self, f"_{self._name}", value)

        @wrap_func_as_method(cls, name=name)
        def getter(that_self) -> prop_type:
            value = getattr(that_self, f"_{self._name}")
            if value is None:
                value = default
            return value

        super().__init__(fget=getter, fset=setter, doc=doc)

    @property
    def allow_none(self) -> bool:
        return self._allow_none

    @property
    def property_type(self) -> type:
        return self._type

    def as_dict_value(self, value):
        if value is None:
            return None
        elif self._is_parameter_value:
            return value.full_label
        elif self._is_parameter_list:
            return [v.full_label for v in value]
        elif self._is_parameter_dict:
            return {k: v.full_label for k, v in value.items()}
        return value

    def validate(self, value: Any, model: Model, parameters: ParameterGroup = None) -> list[str]:

        if value is None and self.allow_none:
            return []

        missing_model = []
        if self._name in model.model_items:
            items = getattr(model, self._name)

            if isinstance(value, list):
                for item in value:
                    if item not in items:
                        missing_model.append((self._name, item))
            elif isinstance(value, dict):
                for item in value.values():
                    if item not in items:
                        missing_model.append((self._name, item))
            elif value not in items:
                missing_model.append((self._name, value))
        missing_model = [
            f"Missing Model Item: '{name}'['{label}']" for name, label in missing_model
        ]

        missing_parameters = []
        if parameters is not None and self._is_parameter and value is not None:
            if self._is_parameter_value:
                if not parameters.has(value.full_label):
                    missing_parameters.append(value.full_label)
            elif self._is_parameter_list:
                for item in value:
                    if not parameters.has(item.full_label):
                        missing_parameters.append(item.full_label)
            elif self._is_parameter_dict:
                for item in value.values():
                    if not parameters.has(item.full_label):
                        missing_parameters.append(item.full_label)
        missing_parameters = [f"Missing Parameter: '{p}'" for p in missing_parameters]

        return missing_model + missing_parameters

    def fill(self, value: Any, model: Model, parameter: ParameterGroup) -> Any:

        if value is None:
            return None

        if self._is_parameter:

            if self._is_parameter_value:
                value.set_from_group(parameter)

            elif self._is_parameter_list:
                for v in value:
                    v.set_from_group(parameter)

            elif self._is_parameter_dict:
                for v in value.values():
                    v.set_from_group(parameter)

        elif hasattr(model, self._name):
            if isinstance(value, list):
                value = [getattr(model, self._name)[v].fill(model, parameter) for v in value]
            elif isinstance(value, dict):
                value = {
                    k: getattr(model, self._name)[v].fill(model, parameter)
                    for (k, v) in value.items()
                }
            elif not isinstance(value, bool):
                value = getattr(model, self._name)[value].fill(model, parameter)

        return value

    def get_parameters(self, value: Any) -> list[str]:
        if value is None:
            return []
        elif self._is_parameter_value:
            return [value.full_label]
        elif self._is_parameter_list:
            return [v.full_label for v in value]
        elif self._is_parameter_dict:
            return [v.full_label for v in value.values()]
        return []

    def _determine_if_parameter(self, type):
        self._is_parameter_value = type is Parameter
        self._is_parameter_list = (
            hasattr(type, "__origin__")
            and issubclass(type.__origin__, List)
            and type.__args__[0] is Parameter
        )
        self._is_parameter_dict = (
            hasattr(type, "__origin__")
            and issubclass(type.__origin__, Dict)
            and type.__args__[1] is Parameter
        )
        self._is_parameter = (
            self._is_parameter_value or self._is_parameter_list or self._is_parameter_dict
        )
