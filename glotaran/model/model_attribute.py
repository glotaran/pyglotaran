"""This package contains the glotaran model item decorator."""

import copy
import typing

import glotaran
from glotaran.parameter import Parameter, ParameterGroup

from .model_property import ModelProperty
from .util import wrap_func_as_method


def model_attribute(properties={},
                    has_type=False,
                    no_label=False):
    """The model_attribute decorator adds the given attributes to the class and applies
    the `dataclass` on it. Further it adds `from_dict` and `from_list`
    classmethods for serialization. Also a `validate_model` and
    `validate_parameter` method is created.

    Classes with the glotaran_model_attribute decorator intended to be used in
    glotaran models.

    Parameters
    ----------
    attributes: Dict[str, type]
        (default value = {})
        A dictonary of attribute names and types.
    has_type: bool
        (default value = False)
        If true, a type attribute will added. Used for model attributes, which
        can have more then one type (e.g. compartment constraints)
    no_label: bool
        (default value = False)
        If true no label attribute will be added. Use for nested items.
    """
    def decorator(cls):

        setattr(cls, '_glotaran_has_label', not no_label)
        setattr(cls, '_glotaran_model_attribute', True)

        # store for later sanity checking
        if not hasattr(cls, '_glotaran_properties'):
            setattr(cls, '_glotaran_properties', [])
            if not no_label:
                doc = f'The label of {cls.__name__} item.'
                prop = ModelProperty(cls, 'label', str, doc, None, False)
                setattr(cls, 'label', prop)
                getattr(cls, '_glotaran_properties').append('label')
            if has_type:
                doc = f'The type string of {cls.__name__}.'
                prop = ModelProperty(cls, 'type', str, doc, None, False)
                setattr(cls, 'type', prop)
                getattr(cls, '_glotaran_properties').append('type')

        else:
            setattr(cls, '_glotaran_properties',
                    [attr for attr in getattr(cls, '_glotaran_properties')])

        for name, options in properties.items():
            if not isinstance(options, dict):
                options = {'type': options}
            prop = ModelProperty(cls, name,
                                 options.get('type'),
                                 options.get('doc', f'{name}'),
                                 options.get('default', None),
                                 options.get('allow_none', False)
                                 )
            setattr(cls, name, prop)
            getattr(cls, '_glotaran_properties').append(name)

        init = _create_init_func(cls)
        setattr(cls, '__init__', init)

        from_dict = _create_from_dict_func(cls)
        setattr(cls, 'from_dict', from_dict)

        from_list = _create_from_list_func(cls)
        setattr(cls, 'from_list', from_list)

        validate = _create_validation_func(cls)
        setattr(cls, 'validate', validate)

        fill = _create_fill_func(cls)
        setattr(cls, 'fill', fill)

        mprint = _create_mprint_func(cls)
        setattr(cls, 'mprint', mprint)

        return cls

    return decorator


def model_attribute_typed(types: typing.Dict[str, any] = {}, no_label=False):
    """The model_attribute_typed decorator adds attributes to the class to enable
    the glotaran model parser to infer the correct class an item when there
    are multiple variants. See package glotaran.model.compartment_constraints
    for an example.

    Parameters
    ----------
    types : dict(str, any)
        A dictonary of types and options.
    """

    def decorator(cls):

        setattr(cls, '_glotaran_model_attribute', True)
        setattr(cls, '_glotaran_model_attribute_typed', True)
        setattr(cls, '_glotaran_model_attribute_types', types)

        setattr(cls, '_glotaran_has_label', not no_label)

        return cls

    return decorator


def _create_init_func(cls):

    @classmethod
    @wrap_func_as_method(cls)
    def __init__(self):
        for attr in self._glotaran_properties:
            setattr(self, f'_{attr}', None)
    return __init__


def _create_from_dict_func(cls):

    @classmethod
    @wrap_func_as_method(cls)
    def from_dict(ncls, values: typing.Dict) -> cls:
        f"""Creates an instance of {cls.__name__} from a dictonary of values.

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
                    raise Exception(f"Missing Property '{name}' For Item '{ncls.__name__}'")
            else:
                setattr(item, name, values[name])

        return item
    return from_dict


def _create_from_list_func(cls):

    @classmethod
    @wrap_func_as_method(cls)
    def from_list(ncls, values: typing.List) -> cls:
        f"""Creates an instance of {cls.__name__} from a list of values. Intended only for internal use.

        Parameters
        ----------
        values :
            A list of values.
        """
        item = ncls()
        if len(values) is not len(ncls._glotaran_properties):
            raise Exception(f"To few or much parameters for '{ncls.__name__}'"
                            f"\nGot: {values}\nWant: {ncls._glotaran_properties}")

        for i, name in enumerate(ncls._glotaran_properties):
            setattr(item, name, values[i])

        return item
    return from_list


def _create_validation_func(cls):

    @wrap_func_as_method(cls)
    def validate(self, model, parameter=None) -> typing.List[str]:
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
            errors += prop.validate(value, model, parameter)
        return errors

    return validate


def _create_fill_func(cls):

    @wrap_func_as_method(cls)
    def fill(self, model: 'glotaran.model.BaseModel', parameter: ParameterGroup) -> cls:
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
            value = prop.fill(value, model, parameter)
            setattr(item, name, value)
        return item
    return fill


def _create_mprint_func(cls):

    @wrap_func_as_method(cls, name='mprint')
    def mprint_item(self, parameter: ParameterGroup = None, initial: ParameterGroup = None) -> str:
        f'''Returns a string with the {cls.__name__} formatted in markdown.'''

        s = "\n"
        if self._glotaran_has_label:
            s = f"**{self.label}**"

            if hasattr(self, 'type'):
                s += f" ({self.type})"
            s += ":\n"
        elif hasattr(self, 'type'):
            s = f"**{self.type}**:\n"

        attrs = []
        for name in self._glotaran_properties:
            value = getattr(self, name)
            if not value:
                continue
            a = f"* *{name.replace('_', ' ').title()}*: "

            def format_parameter(param):
                s = f"{param.full_label}"
                if parameter:
                    p = parameter.get(param.full_label)
                    s += f": **{p.value:.5e}**"
                    if p.vary:
                        err = p.stderr if p.stderr else 0
                        s += f" *(StdErr: {err:.0e}"
                        if initial:
                            i = initial.get(param.full_label)
                            s += f" ,initial: {i.value:.5e}"
                        s += ")*"
                    else:
                        s += " *(fixed)*"
                return s

            if isinstance(value, Parameter):
                a += format_parameter(value)
            elif isinstance(value, list) and all(isinstance(v, Parameter) for v in value):
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
