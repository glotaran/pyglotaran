"""The model attribute decorator."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from glotaran.parameter import Parameter

from .property import ModelProperty
from .util import wrap_func_as_method

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable

    from glotaran.parameter import ParameterGroup

    from .base_model import Model


def model_attribute(
    properties: Any | dict[str, dict[str, Any]] = {},
    has_type: bool = False,
    no_label: bool = False,
) -> Callable:
    """The `@model_attribute` decorator adds the given properties to the class. Further it adds
    classmethods for deserialization, validation and printing.

    By default, a `label` property is added.

    The `properties` dictionary contains the name of the properties as keys. The values must be
    either a `type` or dictionary with the following values:

    * type: a `type` (required)
    * doc: a string for documentation (optional)
    * default: a default value (optional)
    * allow_none: if `True`, the property can be set to None (optional)

    Classes with the `model_attribute` decorator intended to be used in glotaran models.

    Parameters
    ----------
    properties :
        A dictionary of property names and options.
    has_type:
        If true, a type property will added. Used for model attributes, which
        can have more then one type.
    no_label:
        If true no label property will be added.
    """

    def decorator(cls):

        setattr(cls, "_glotaran_has_label", not no_label)
        setattr(cls, "_glotaran_model_attribute", True)

        # store for later sanity checking
        if not hasattr(cls, "_glotaran_properties"):
            setattr(cls, "_glotaran_properties", [])
            if not no_label:
                doc = f"The label of {cls.__name__} item."
                prop = ModelProperty(cls, "label", str, doc, None, False)
                setattr(cls, "label", prop)
                getattr(cls, "_glotaran_properties").append("label")
            if has_type:
                doc = f"The type string of {cls.__name__}."
                prop = ModelProperty(cls, "type", str, doc, None, False)
                setattr(cls, "type", prop)
                getattr(cls, "_glotaran_properties").append("type")

        else:
            setattr(
                cls,
                "_glotaran_properties",
                [attr for attr in getattr(cls, "_glotaran_properties")],
            )

        for name, options in properties.items():
            if not isinstance(options, dict):
                options = {"type": options}
            prop = ModelProperty(
                cls,
                name,
                options.get("type"),
                options.get("doc", f"{name}"),
                options.get("default", None),
                options.get("allow_none", False),
            )
            setattr(cls, name, prop)
            if name not in getattr(cls, "_glotaran_properties"):
                getattr(cls, "_glotaran_properties").append(name)

        init = _create_init_func(cls)
        setattr(cls, "__init__", init)

        from_dict = _create_from_dict_func(cls)
        setattr(cls, "from_dict", from_dict)

        from_list = _create_from_list_func(cls)
        setattr(cls, "from_list", from_list)

        validate = _create_validation_func(cls)
        setattr(cls, "validate", validate)

        get_state = _create_get_state_func(cls)
        setattr(cls, "__getstate__", get_state)

        set_state = _create_set_state_func(cls)
        setattr(cls, "__setstate__", set_state)

        fill = _create_fill_func(cls)
        setattr(cls, "fill", fill)

        mprint = _create_mprint_func(cls)
        setattr(cls, "mprint", mprint)

        return cls

    return decorator


def model_attribute_typed(types: dict[str, Any], no_label=False):
    """The model_attribute_typed decorator adds attributes to the class to enable
    the glotaran model parser to infer the correct class for an item when there
    are multiple variants.

    Parameters
    ----------
    types :
        A dictionary of types and options.
    no_label:
        If `True` no label property will be added.
    """

    def decorator(cls):

        setattr(cls, "_glotaran_model_attribute", True)
        setattr(cls, "_glotaran_model_attribute_typed", True)
        setattr(cls, "_glotaran_model_attribute_types", types)

        add_type = _create_add_type_func(cls)
        setattr(cls, "add_type", add_type)

        setattr(cls, "_glotaran_has_label", not no_label)

        return cls

    return decorator


def _create_add_type_func(cls):
    @classmethod
    @wrap_func_as_method(cls)
    def add_type(cls, type_name: str, attribute_type: type):
        getattr(cls, "_glotaran_model_attribute_types")[type_name] = attribute_type

    return add_type


def _create_init_func(cls):
    @classmethod
    @wrap_func_as_method(cls)
    def __init__(self):
        for attr in self._glotaran_properties:
            setattr(self, f"_{attr}", None)

    return __init__


def _create_from_dict_func(cls):
    @classmethod
    @wrap_func_as_method(cls)
    def from_dict(ncls, values: dict) -> cls:
        f"""Creates an instance of {cls.__name__} from a dictionary of values.

        Intended only for internal use.

        Parameters
        ----------
        values :
            A list of values.
        """
        item = ncls()

        for name in ncls._glotaran_properties:
            if name not in values:
                if not getattr(ncls, name).allow_none and getattr(item, name) is None:
                    raise ValueError(f"Missing Property '{name}' For Item '{ncls.__name__}'")
            else:
                setattr(item, name, values[name])

        return item

    return from_dict


def _create_from_list_func(cls):
    @classmethod
    @wrap_func_as_method(cls)
    def from_list(ncls, values: list) -> cls:
        f"""Creates an instance of {cls.__name__} from a list of values. Intended only for internal use.

        Parameters
        ----------
        values :
            A list of values.
        """
        item = ncls()
        if len(values) is not len(ncls._glotaran_properties):
            raise ValueError(
                f"To few or much parameters for '{ncls.__name__}'"
                f"\nGot: {values}\nWant: {ncls._glotaran_properties}"
            )

        for i, name in enumerate(ncls._glotaran_properties):
            setattr(item, name, values[i])

        return item

    return from_list


def _create_validation_func(cls):
    @wrap_func_as_method(cls)
    def validate(self, model: Model, parameters=None) -> list[str]:
        f"""Creates a list of parameters needed by this instance of {cls.__name__} not present in a
        set of parameters.

        Parameters
        ----------
        model :
            The model to validate.
        parameter :
            The parameter to validate.
        missing :
            A list the missing will be appended to.
        """
        errors = []
        for name in self._glotaran_properties:
            prop = getattr(self.__class__, name)
            value = getattr(self, name)
            errors += prop.validate(value, model, parameters)
        return errors

    return validate


def _create_fill_func(cls):
    @wrap_func_as_method(cls)
    def fill(self, model: Model, parameters: ParameterGroup) -> cls:
        """Returns a copy of the {cls._name} instance with all members which are Parameters are
        replaced by the value of the corresponding parameter in the parameter group.

        Parameters
        ----------
        model :
            A glotaran model.
        parameter : ParameterGroup
            The parameter group to fill from.
        """
        item = copy.deepcopy(self)
        for name in self._glotaran_properties:
            prop = getattr(self.__class__, name)
            value = getattr(self, name)
            value = prop.fill(value, model, parameters)
            setattr(item, name, value)
        return item

    return fill


def _create_get_state_func(cls):
    @wrap_func_as_method(cls)
    def get_state(self) -> cls:
        return tuple(getattr(self, name) for name in self._glotaran_properties)

    return get_state


def _create_set_state_func(cls):
    @wrap_func_as_method(cls)
    def set_state(self, state) -> cls:
        for i, name in enumerate(self._glotaran_properties):
            setattr(self, name, state[i])

    return set_state


def _create_mprint_func(cls):
    @wrap_func_as_method(cls, name="mprint")
    def mprint_item(
        self, parameters: ParameterGroup = None, initial_parameters: ParameterGroup = None
    ) -> str:
        f"""Returns a string with the {cls.__name__} formatted in markdown."""

        s = "\n"
        if self._glotaran_has_label:
            s = f"**{self.label}**"

            if hasattr(self, "type"):
                s += f" ({self.type})"
            s += ":\n"
        elif hasattr(self, "type"):
            s = f"**{self.type}**:\n"

        attrs = []
        for name in self._glotaran_properties:
            value = getattr(self, name)
            if value is None:
                continue
            a = f"* *{name.replace('_', ' ').title()}*: "

            def format_parameter(param):
                s = f"{param.full_label}"
                if parameters is not None:
                    p = parameters.get(param.full_label)
                    s += f": **{p.value:.5e}**"
                    if p.vary:
                        err = p.standard_error or 0
                        s += f" *(StdErr: {err:.0e}"
                        if initial_parameters is not None:
                            i = initial_parameters.get(param.full_label)
                            s += f" ,initial: {i.value:.5e}"
                        s += ")*"
                    else:
                        s += " *(fixed)*"
                return s

            if isinstance(value, Parameter):
                a += format_parameter(value)
            elif isinstance(value, list) and all([isinstance(v, Parameter) for v in value]):
                a += f"[{', '.join([format_parameter(v) for v in value])}]"
            elif isinstance(value, dict):
                a += "\n"
                for k, v in value.items():
                    a += f"  * *{k}*: "
                    if isinstance(v, Parameter):
                        a += format_parameter(v)
                    else:
                        a += f"{v}"
                    a += "\n"
            else:
                a += f"{value}"
            attrs.append(a)
        s += "\n".join(attrs)
        return s

    return mprint_item
