"""The model item decorator."""
from __future__ import annotations

import copy
from textwrap import indent
from typing import TYPE_CHECKING
from typing import Callable
from typing import List
from typing import Type

from glotaran.model.property import ModelProperty
from glotaran.model.util import wrap_func_as_method
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from typing import Any

    from glotaran.model.model import Model
    from glotaran.parameter import ParameterGroup

    Validator = Callable[
        [Type[object], Type[Model]],
        List[str],
    ]

    ValidatorParameter = Callable[
        [Type[object], Type[Model], Type[ParameterGroup]],
        List[str],
    ]


def model_item(
    properties: None | dict[str, dict[str, Any]] = None,
    has_type: bool = False,
    has_label: bool = True,
) -> Callable:
    """The `@model_item` decorator adds the given properties to the class. Further it adds
    classmethods for deserialization, validation and printing.

    By default, a `label` property is added.

    The `properties` dictionary contains the name of the properties as keys. The values must be
    either a `type` or dictionary with the following values:

    * type: a `type` (required)
    * doc: a string for documentation (optional)
    * default: a default value (optional)
    * allow_none: if `True`, the property can be set to None (optional)

    Classes with the `model_item` decorator intended to be used in glotaran models.

    Parameters
    ----------
    properties :
        A dictionary of property names and options.
    has_type :
        If true, a type property will added. Used for model attributes, which
        can have more then one type.
    has_label :
        If false no label property will be added.
    """

    if properties is None:
        properties = {}

    def decorator(cls):

        setattr(cls, "_glotaran_has_label", has_label)
        setattr(cls, "_glotaran_model_item", True)

        # store for later sanity checking
        if not hasattr(cls, "_glotaran_properties"):
            setattr(cls, "_glotaran_properties", [])
            if has_label:
                doc = f"The label of {cls.__name__} item."
                prop = ModelProperty(cls, "label", str, doc, None, False)
                setattr(cls, "label", prop)
                getattr(cls, "_glotaran_properties").append("label")
            if has_type:
                doc = f"The type string of {cls.__name__}."
                prop = ModelProperty(cls, "type", str, doc, None, True)
                setattr(cls, "type", prop)
                getattr(cls, "_glotaran_properties").append("type")

        else:
            setattr(
                cls,
                "_glotaran_properties",
                list(getattr(cls, "_glotaran_properties")),
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

        validators = _get_validators(cls)
        setattr(cls, "_glotaran_validators", validators)

        init = _init_factory(cls)
        setattr(cls, "__init__", init)

        from_dict = _from_dict_factory(cls)
        setattr(cls, "from_dict", from_dict)

        validate = _validation_factory(cls)
        setattr(cls, "validate", validate)

        as_dict = _as_dict_factory(cls)
        setattr(cls, "as_dict", as_dict)

        get_state = _get_state_factory(cls)
        setattr(cls, "__getstate__", get_state)

        set_state = _set_state_factory(cls)
        setattr(cls, "__setstate__", set_state)

        fill = _fill_factory(cls)
        setattr(cls, "fill", fill)

        markdown = _markdown_factory(cls)
        setattr(cls, "markdown", markdown)

        get_parameter_labels = _get_parameter_labels_factory(cls)
        setattr(cls, "get_parameter_labels", get_parameter_labels)

        return cls

    return decorator


def model_item_typed(
    *,
    types: dict[str, Any],
    has_label: bool = True,
    default_type: str = None,
):
    """The model_item_typed decorator adds attributes to the class to enable
    the glotaran model parser to infer the correct class for an item when there
    are multiple variants.

    Parameters
    ----------
    types :
        A dictionary of types and options.
    has_label:
        If `False` no label property will be added.
    """

    def decorator(cls):

        setattr(cls, "_glotaran_model_item", True)
        setattr(cls, "_glotaran_model_item_typed", True)
        setattr(cls, "_glotaran_model_item_types", types)
        setattr(cls, "_glotaran_model_item_default_type", default_type)

        get_default_type = _get_default_type_factory(cls)
        setattr(cls, "get_default_type", get_default_type)

        add_type = _add_type_factory(cls)
        setattr(cls, "add_type", add_type)

        setattr(cls, "_glotaran_has_label", has_label)

        return cls

    return decorator


def model_item_validator(need_parameter: bool):
    """The model_item_validator marks a method of a model item as validation function"""

    def decorator(method: Validator | ValidatorParameter):
        setattr(method, "_glotaran_validator", need_parameter)
        return method

    return decorator


def _get_validators(cls):
    return {
        method: getattr(getattr(cls, method), "_glotaran_validator")
        for method in dir(cls)
        if hasattr(getattr(cls, method), "_glotaran_validator")
    }


def _get_default_type_factory(cls):
    @classmethod
    @wrap_func_as_method(cls)
    def get_default_type(cls) -> str:
        return getattr(cls, "_glotaran_model_item_default_type")

    return get_default_type


def _add_type_factory(cls):
    @classmethod
    @wrap_func_as_method(cls)
    def add_type(cls, type_name: str, attribute_type: type):
        getattr(cls, "_glotaran_model_item_types")[type_name] = attribute_type

    return add_type


def _init_factory(cls):
    @classmethod
    @wrap_func_as_method(cls)
    def __init__(self):
        for attr in self._glotaran_properties:
            setattr(self, f"_{attr}", None)

    return __init__


def _from_dict_factory(cls):
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
            if name in values:
                value = values[name]
                prop = getattr(item.__class__, name)
                if prop.glotaran_property_type == float:
                    value = float(value)
                elif prop.glotaran_property_type == int:
                    value = int(value)
                setattr(item, name, value)

            elif not getattr(ncls, name).glotaran_allow_none and getattr(item, name) is None:
                raise ValueError(f"Missing Property '{name}' For Item '{ncls.__name__}'")
        return item

    return from_dict


def _validation_factory(cls):
    @wrap_func_as_method(cls)
    def validate(self, model: Model, parameters: ParameterGroup | None = None) -> list[str]:
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
        problems = []
        for name in self._glotaran_properties:
            prop = getattr(self.__class__, name)
            value = getattr(self, name)
            problems += prop.glotaran_validate(value, model, parameters)
        for validator, need_parameter in self._glotaran_validators.items():
            if need_parameter:
                if parameters is not None:
                    problems += getattr(self, validator)(model, parameters)
            else:
                problems += getattr(self, validator)(model)

        return problems

    return validate


def _as_dict_factory(cls):
    @wrap_func_as_method(cls)
    def as_dict(self) -> dict:
        return {
            name: getattr(self.__class__, name).glotaran_replace_parameter_with_labels(
                getattr(self, name)
            )
            for name in self._glotaran_properties
            if name != "label" and getattr(self, name) is not None
        }

    return as_dict


def _fill_factory(cls):
    @wrap_func_as_method(cls)
    def fill(self, model: Model, parameters: ParameterGroup) -> cls:
        f"""Returns a copy of the {cls.__name__} instance with all members which are Parameters are
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
            value = prop.glotaran_fill(value, model, parameters)
            setattr(item, name, value)
        return item

    return fill


def _get_state_factory(cls):
    @wrap_func_as_method(cls)
    def get_state(self) -> cls:
        return tuple(getattr(self, name) for name in self._glotaran_properties)

    return get_state


def _set_state_factory(cls):
    @wrap_func_as_method(cls)
    def set_state(self, state) -> cls:
        for i, name in enumerate(self._glotaran_properties):
            setattr(self, name, state[i])

    return set_state


def _markdown_factory(cls):
    @wrap_func_as_method(cls, name="markdown")
    def mprint_item(
        self, all_parameters: ParameterGroup = None, initial_parameters: ParameterGroup = None
    ) -> MarkdownStr:
        f"""Returns a string with the {cls.__name__} formatted in markdown."""

        md = "\n"
        if self._glotaran_has_label:
            md = f"**{self.label}**"
            if hasattr(self, "type"):
                md += f" ({self.type})"
            md += ":\n"

        elif hasattr(self, "type"):
            md = f"**{self.type}**:\n"

        for name in self._glotaran_properties:
            prop = getattr(self.__class__, name)
            value = getattr(self, name)
            if value is None:
                continue
            property_md = indent(
                f"* *{name.replace('_', ' ').title()}*: "
                f"{prop.glotaran_value_as_markdown(value,all_parameters, initial_parameters)}\n",
                "  ",
            )

            md += property_md

        return MarkdownStr(md)

    return mprint_item


def _get_parameter_labels_factory(cls):
    @wrap_func_as_method(cls, name="get_parameter_labels")
    def get_parameter_labels(self) -> list[str]:

        parameter_labels = []

        for name in self._glotaran_properties:
            prop = getattr(self.__class__, name)
            value = getattr(self, name)
            parameter_labels += prop.glotaran_get_parameter_labels(value)

        return parameter_labels

    return get_parameter_labels
