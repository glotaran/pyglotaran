"""The model property class."""

import typing

from glotaran.parameter import Parameter

from .util import wrap_func_as_method


class ModelProperty(property):
    def __init__(self, cls, name, prop_type, doc, default, allow_none):

        self._name = name
        self._allow_none = allow_none
        self._determine_if_parameter(prop_type)

        set_type = prop_type if not self._is_parameter else typing.Union[str, prop_type]

        @wrap_func_as_method(cls, name=name)
        def setter(that_self, value: set_type):
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

    def validate(self, value, model, parameters=None) -> typing.List[str]:

        if value is None and self.allow_none:
            return []

        missing_model = []
        if hasattr(model, f"set_{self._name}") or hasattr(model, f"add_{self._name}"):
            attr = getattr(model, self._name)

            if isinstance(value, list):
                for item in value:
                    if item not in attr:
                        missing_model.append((self._name, item))
            elif isinstance(value, dict):
                for item in value.values():
                    if item not in attr:
                        missing_model.append((self._name, item))
            else:
                if value not in attr:
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

    def fill(self, value, model, parameter):

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
            else:
                value = getattr(model, self._name)[value].fill(model, parameter)

        return value

    def _determine_if_parameter(self, type):
        self._is_parameter_value = type is Parameter
        self._is_parameter_list = (
            hasattr(type, "__origin__")
            and issubclass(type.__origin__, typing.List)
            and type.__args__[0] is Parameter
        )
        self._is_parameter_dict = (
            hasattr(type, "__origin__")
            and issubclass(type.__origin__, typing.Dict)
            and type.__args__[1] is Parameter
        )
        self._is_parameter = (
            self._is_parameter_value or self._is_parameter_list or self._is_parameter_dict
        )
