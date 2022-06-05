"""This module holds the model property class."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence
from typing import TypeVar

from glotaran.model.util import get_subtype
from glotaran.model.util import is_mapping_type
from glotaran.model.util import is_scalar_type
from glotaran.model.util import is_sequence_type
from glotaran.model.util import wrap_func_as_method
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from glotaran.model.model import Model


ParameterOrLabel = TypeVar("ParameterOrLabel", str, Parameter)


class ModelProperty(property):
    """ModelProperty is an extension of the property decorator.

    It adds convenience functions for meta programming model items.
    """

    def __init__(
        self, cls: type, name: str, property_type: type, doc: str, default: Any, allow_none: bool
    ):
        """Create a new model property.

        Parameters
        ----------
        cls : type
            The class the property is being attached to.
        name : str
            The name of the property.
        property_type : type
            The type of the property.
        doc : str
            A documentation string of for the property.
        default : Any
            The default value of the property.
        allow_none : bool
            Whether the property is allowed to be None.
        """
        self._name = name
        self._allow_none = allow_none
        self._default = default

        if get_subtype(property_type) is Parameter:
            if is_scalar_type(property_type):
                property_type = ParameterOrLabel  # type: ignore[assignment]
            elif is_sequence_type(property_type):
                property_type = Sequence[ParameterOrLabel]
            elif is_mapping_type(property_type):
                property_type = Mapping[
                    property_type.__args__[0], ParameterOrLabel  # type: ignore[name-defined]
                ]

        self._type = property_type

        super().__init__(
            fget=_model_property_getter_factory(cls, self),
            fset=_model_property_setter_factory(cls, self),
            doc=doc,
        )

    @property
    def glotaran_allow_none(self) -> bool:
        """Check if the property is allowed to be None.

        Returns
        -------
        bool
            Whether the property is allowed to be None.
        """
        return self._allow_none

    @property
    def glotaran_property_type(self) -> type:
        """Get the type of the property.

        Returns
        -------
        type
            The type of the property.
        """
        return self._type

    @property
    def glotaran_is_scalar_property(self) -> bool:
        """Check if the type is scalar.

        Scalar means the type is neither a sequence nor a mapping.

        Returns
        -------
        bool
            Whether the type is scalar.
        """
        return is_scalar_type(self._type)

    @property
    def glotaran_is_sequence_property(self) -> bool:
        """Check if the type is a sequence.

        Returns
        -------
        bool
            Whether the type is a sequence.
        """
        return is_sequence_type(self._type)

    @property
    def glotaran_is_mapping_property(self) -> bool:
        """Check if the type is mapping.

        Returns
        -------
        bool
            Whether the type is a mapping.
        """
        return is_mapping_type(self._type)

    @property
    def glotaran_property_subtype(self) -> type:
        """Get the subscribed type.

        If the type is scalar, the type itself will be returned. If the type is a mapping,
        the value type will be returned.

        Returns
        -------
        type
            The subscribed type.
        """
        return get_subtype(self._type)

    @property
    def glotaran_is_parameter_property(self) -> bool:
        """Check if the subtype is parameter.

        Returns
        -------
        bool
            Whether the subtype is parameter.
        """
        return self.glotaran_property_subtype is ParameterOrLabel

    def glotaran_replace_parameter_with_labels(self, value: Any) -> Any:
        """Replace parameter values with their full label.

        A convenience function for serialization.

        Parameters
        ----------
        value : Any
            The value to replace.

        Returns
        -------
        Any
            The value with parameters replaced by their labels.
        """
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
        """Validate a value against a model and optionally against parameters.

        Parameters
        ----------
        value : Any
            The value to validate.
        model : Model
            The model to validate against.
        parameters : ParameterGroup
            The parameters to validate against.

        Returns
        -------
        list[str]
            A list of human readable list of messages of problems.
        """
        if value is None:
            if self.glotaran_allow_none:
                return []
            else:
                return [f"Property '{self._name}' is none but not allowed to be none."]

        missing_model: list[tuple[str, str]] = []
        if self._name in model.model_items:
            items = getattr(model, self._name)

            if self.glotaran_is_sequence_property:
                missing_model.extend((self._name, item) for item in value if item not in items)
            elif self.glotaran_is_mapping_property:
                missing_model.extend(
                    (self._name, item) for item in value.values() if item not in items
                )
            elif value not in items:
                missing_model.append((self._name, value))
        missing_model_messages = [
            f"Missing Model Item: '{name}'['{label}']" for name, label in missing_model
        ]

        missing_parameters: list[str] = []
        if parameters is not None and self.glotaran_is_parameter_property:
            wanted = value
            if self.glotaran_is_scalar_property:
                wanted = [wanted]
            elif self.glotaran_is_mapping_property:
                wanted = wanted.values()
            missing_parameters.extend(
                parameter.full_label
                for parameter in wanted
                if not parameters.has(parameter.full_label)
            )
        missing_parameters_messages = [f"Missing Parameter: '{p}'" for p in missing_parameters]

        return missing_model_messages + missing_parameters_messages

    def glotaran_fill(self, value: Any, model: Model, parameter: ParameterGroup) -> Any:
        """Fill a property with items from a model and parameters.

        This replaces model item labels with the actual items and sets the parameter values.

        Parameters
        ----------
        value : Any
            The property value.
        model : Model
            The model to fill in.
        parameter : ParameterGroup
            The parameters to fill in.

        Returns
        -------
        Any
            The filled value.
        """
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
        """Get a markdown representation of the property.

        Parameters
        ----------
        value : Any
            The property value.
        all_parameters : ParameterGroup | None
            A parameter group containing the whole parameter set (used for expression lookup).
        initial_parameters : ParameterGroup | None
            The initial parameter.

        Returns
        -------
        MarkdownStr
            The property as markdown string.
        """
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
        """Format a value to string.

        Parameters
        ----------
        value : Any
            The value to format.
        all_parameters : ParameterGroup | None
            A parameter group containing the whole parameter set (used for expression lookup).
        initial_parameters : ParameterGroup | None
            The initial parameter.

        Returns
        -------
        str
            The formatted value.
        """
        return (
            value.markdown(all_parameters, initial_parameters)
            if self.glotaran_is_parameter_property
            else str(value)
        )

    def glotaran_get_parameter_labels(self, value: Any) -> list[str]:
        """Get a list of all parameter labels if the property is parameter.

        Parameters
        ----------
        value : Any
            The value of the property.

        Returns
        -------
        list[str]
            The list of full parameter labels.
        """
        if value is None or not self.glotaran_is_parameter_property:
            return []
        elif self.glotaran_is_sequence_property:
            return [v.full_label for v in value]
        elif self.glotaran_is_mapping_property:
            return [v.full_label for v in value.values()]
        return [value.full_label]


def _model_property_getter_factory(cls: type, model_property: ModelProperty) -> Callable:
    """Create a getter function for model property.

    Parameters
    ----------
    cls: type
        The class to create the getter for.
    model_property : ModelProperty
        The property to create the getter for.

    Returns
    -------
    Callable
        The created getter.
    """

    @wrap_func_as_method(cls, name=model_property._name)
    def getter(self) -> model_property.glotaran_property_type:  # type: ignore[name-defined]
        value = getattr(self, f"_{model_property._name}")
        if value is None:
            value = model_property._default
        return value

    return getter


def _model_property_setter_factory(cls: type, model_property: ModelProperty):
    """Create a setter function for model property.

    Parameters
    ----------
    cls: type
        The class to create the setter for.
    model_property : ModelProperty
        The property to create the setter for.

    Returns
    -------
    Callable
        The created setter.
    """

    @wrap_func_as_method(cls, name=model_property._name)
    def setter(self, value: model_property.glotaran_property_type):  # type: ignore[name-defined]
        if value is None and not model_property._allow_none:
            raise ValueError(
                f"Property '{model_property._name}' of '{cls.__name__}' "
                "is not allowed to set to None."
            )
        if value is not None and model_property.glotaran_is_parameter_property:
            if model_property.glotaran_is_scalar_property and not isinstance(value, Parameter):
                value = Parameter(full_label=str(value))
            elif model_property.glotaran_is_sequence_property and all(
                not isinstance(v, Parameter) for v in value
            ):
                value = [Parameter(full_label=str(v)) for v in value]
            elif model_property.glotaran_is_mapping_property and all(
                not isinstance(v, Parameter) for v in value.values()
            ):
                value = {k: Parameter(full_label=str(v)) for k, v in value.items()}
        setattr(self, f"_{model_property._name}", value)

    return setter
