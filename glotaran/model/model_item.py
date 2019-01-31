"""This package contains the glotaran model item decorator."""

import typing
import inspect
from dataclasses import dataclass, replace

import glotaran
from glotaran.parameter import Parameter, ParameterGroup

from .model_item_validator import Validator
from .util import item_or_list_to_arg, wrap_func_as_method


def model_item(attributes={},
               has_type=False,
               no_label=False):
    """The model_item decorator adds the given attributes to the class and applies
    the `dataclass` on it. Further it adds `from_dict` and `from_list`
    classmethods for serialization. Also a `validate_model` and
    `validate_parameter` method is created.

    Classes with the glotaran_model_item decorator intended to be used in
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

        # we don't leverage pythons default param mechanism, since this breaks
        # inheritence of items
        if not hasattr(cls, '_glotaran_attribute_defaults'):
            setattr(cls, '_glotaran_attribute_defaults', {})
        else:
            setattr(cls, '_glotaran_attribute_defaults',
                    getattr(cls, '_glotaran_attribute_defaults').copy())

        setattr(cls, '_glotaran_has_label', not no_label)

        # Set annotations for autocomplete and doc
        annotations = {}
        if not no_label:
            annotations['label'] = str
        if has_type:
            annotations['type'] = str
        for name, item_class in attributes.items():
            if isinstance(item_class, dict):
                item_dict = item_class
                item_class = item_dict.get('type', str)
                if 'default' in item_dict:
                    default = item_dict.get('default')
                    getattr(cls, '_glotaran_attribute_defaults')[name] = default
            else:
                attributes[name] = {'type': item_class}
            annotations[name] = item_class
        for name in annotations:
            setattr(cls, name, None)

        setattr(cls, '__annotations__', annotations)

        # We turn it into dataclass to get automagic inits
        cls = dataclass(cls)

        # for nesting
        setattr(cls, '_glotaran_model_item', True)

        # store for later sanity checking
        if not hasattr(cls, '_glotaran_attributes'):
            setattr(cls, '_glotaran_attributes', {})
        else:
            setattr(cls, '_glotaran_attributes',
                    getattr(cls, '_glotaran_attributes').copy())
        for name, opts in attributes.items():
            getattr(cls, '_glotaran_attributes')[name] = opts

        from_dict = _create_from_dict_func(cls)
        setattr(cls, 'from_dict', from_dict)

        from_list = _create_from_list_func(cls)
        setattr(cls, 'from_list', from_list)

        val_model, val_parameter = _create_validation_funcs(cls)
        setattr(cls, 'missing_model_items', val_model)
        setattr(cls, 'missing_parameter', val_parameter)

        fill = _create_fill_func(cls)
        setattr(cls, 'fill', fill)

        mprint = _create_mprint_func(cls)
        setattr(cls, 'mprint', mprint)
        #  setattr(cls, '__str__', functools.wraps(cls.__str__)(mprint))

        return cls

    return decorator


def model_item_typed(types: typing.Dict[str, any] = {}, no_label=False):
    """The model_item_typed decorator adds attributes to the class to enable
    the glotaran model parser to infer the correct class an item when there
    are multiple variants. See package glotaran.model.compartment_constraints
    for an example.

    Parameters
    ----------
    types : dict(str, any)
        A dictonary of types and options.
    """

    def decorator(cls):

        setattr(cls, '_glotaran_model_item', True)
        setattr(cls, '_glotaran_model_item_typed', True)
        setattr(cls, '_glotaran_model_item_types', types)

        setattr(cls, '_glotaran_has_label', not no_label)

        return cls

    return decorator


def _create_from_dict_func(cls):

        @classmethod
        @wrap_func_as_method(cls)
        def from_list(ncls, values: typing.Dict) -> cls:
            f"""Creates an instance of {cls.__name__} from a dictonary of values.

            Intended only for internal use.

            Parameters
            ----------
            values :
                A list of values.
            """
            params = inspect.signature(ncls.__init__).parameters
            args = []
            for name, param in params.items():
                if name == "self":
                    continue
                if name not in values:
                    if name not in getattr(ncls, '_glotaran_attribute_defaults'):
                        raise Exception(f"Missing parameter '{name} for item "
                                        f"'{ncls.__name__}'")
                    args.append(getattr(ncls, '_glotaran_attribute_defaults')[name])
                else:
                    item = values[name]
                    item_class = param.annotation
                    item = item_or_list_to_arg(name, item, item_class)
                    args.append(item)

            return ncls(*args)
        return from_list


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
            names = [n for n in
                     inspect.signature(ncls.__init__).parameters if not n == "self"]
            params = [p for _, p in
                      inspect.signature(ncls.__init__).parameters.items()]
            # params contains 'self'
            params = params[1:]
            if len(values) is not len(params):
                raise Exception(f"To few or much parameters for '{ncls.__name__}'"
                                f"\nGot: {values}\nWant: {names}")

            for i in range(len(values)):
                item_class = params[i].annotation
                values[i] = item_or_list_to_arg(names[i], values[i], item_class)

            return ncls(*values)
        return from_list


def _create_validation_funcs(cls):

    validator = Validator(cls)

    @wrap_func_as_method(cls)
    def missing_model_items(self, model: 'glotaran.model.BaseModel',
                            missing: typing.List[str] = []) -> typing.List[str]:
        f"""Creates a list of model items needed by this instance of {cls.__name__} not present in a model.

        Parameters
        ----------
        model :
            The model to validate.
        missing :
            A list the missing will be appended to.
        """
        return validator.val_model(self, model, missing)

    @wrap_func_as_method(cls)
    def missing_parameter(self, model: 'glotaran.model.BaseModel',
                          parameter: ParameterGroup,
                          missing: typing.List[str] = []) -> typing.List[str]:
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

        return validator.val_parameter(self, model, parameter, missing)

    return missing_model_items, missing_parameter


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

        def convert_list_or_scalar(item):
            if isinstance(item, list):
                return [convert(i) for i in item]
            return convert(item)

        def convert(item, target=None):
            if isinstance(item, dict):
                cp = item.copy()
                for k, v in item.items():
                    if isinstance(v, list):
                        cp[k] = [convert(i) for i in v]
                    else:
                        cp[k] = convert(v)
                item = cp
            elif hasattr(item, "_glotaran_model_item"):
                item = item.fill(model, parameter)
            elif isinstance(item, Parameter):
                item = parameter.get(item.full_label).value
            return item

        def fill_item_or_list(item, attr):
            model_attr = getattr(model, attr)
            if isinstance(item, list):
                return [model_attr[i].fill(model, parameter) for i in item]
            return model_attr[item].fill(model, parameter)

        replaced = {}
        attrs = getattr(self, '_glotaran_attributes')
        for attr, opts in attrs.items():
            item = getattr(self, attr)
            if item is None:
                continue
            if 'target' in opts:
                target = opts['target']
                if isinstance(target, tuple):
                    target = target[1]
                    nitem = {}
                    for k, v in item.items():
                        if target == 'parameter':
                            nitem[k] = convert_list_or_scalar(v)
                        else:
                            nitem[k] = fill_item_or_list(v, target)
                    item = nitem

            elif hasattr(model, attr):
                if attr == 'compartment':
                    continue
                item = fill_item_or_list(item, attr)
            else:
                item = convert_list_or_scalar(item)
            replaced[attr] = item
        return replace(self, **replaced)
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
        for name in self._glotaran_attributes:
            value = getattr(self, name)
            if not value:
                continue
            a = f"* *{name.replace('_', ' ').title()}*: "

            def format_parameter(param):
                s = f"{param.full_label}"
                if parameter:
                    p = parameter.get(param.full_label)
                    s += f": **{p.value}**"
                    if p.vary:
                        err = p.stderr if p.stderr else 0
                        s += f" *(StdErr: {err:.0e}"
                        if initial:
                            i = initial.get(param.full_label)
                            s += f" ,initial: {i.value}"
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
