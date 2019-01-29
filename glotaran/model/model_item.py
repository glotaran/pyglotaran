"""This package contains the glotaran model item decorator."""

from typing import Dict, List
import inspect
import functools
from dataclasses import dataclass, replace

from glotaran.parameter import Parameter, ParameterGroup

from .model_item_validator import Validator


def _is_list_type(item_class):
    return issubclass(item_class.__origin__, List) if \
        hasattr(item_class, '__origin__') else False


def _is_dict(item_class):
    return issubclass(item_class.__origin__, Dict) if \
        hasattr(item_class, '__origin__') else False


def is_item_or_list_of(item_class):
    if not isinstance(item_class, type):
        return False
    islist = _is_list_type(item_class)
    if islist:
        item_class = item_class.__args__[0]
    return hasattr(item_class, "_glotaran_model_item")


def item_or_list_to_arg(name, item, item_class):
    islist = _is_list_type(item_class)
    if islist:
        item_class = item_class.__args__[0]
    if item_class is Parameter:
        if islist:
            item = [Parameter(full_label=i) for i in item]
        else:
            item = Parameter(full_label=item)
    elif hasattr(item_class, "_glotaran_model_item"):
        if not islist:
            item = item_to_param(name, item, item_class)
        else:
            item = [item_to_param(name, i, item_class) for i in item]
    elif _is_dict(item_class):
        v_class = item_class.__args__[1]
        islist = _is_list_type(v_class)
        if islist:
            v_class = v_class.__args__[0]
        if v_class is Parameter:
            for k, v in item.items():
                item[k] = Parameter(full_label=v)
    return item


def item_to_param(name, item, item_class):
    is_typed = hasattr(item_class, "_glotaran_model_item_typed")
    if isinstance(item, dict):
        if is_typed:
            if 'type' not in item:
                raise Exception(f"Missing type for attribute '{name}'")
            item_type = item['type']

            if item_type not in item_class._glotaran_model_item_types:
                raise Exception(f"Unknown type '{item_type}' "
                                f"for attribute '{name}'")
            item_class = \
                item_class._glotaran_model_item_types[item_type]
        return item_class.from_dict(item)
    else:
        if is_typed:
            if len(item) < 2 and len(item) is not 1:
                raise Exception(f"Missing type for attribute '{name}'")
            item_type = item[1] if len(item) is not 1 and \
                hasattr(item_class, 'label') else item[0]

            if item_type not in item_class._glotaran_model_item_types:
                raise Exception(f"Unknown type '{item_type}' "
                                f"for attribute '{name}'")
            item_class = \
                item_class._glotaran_model_item_types[item_type]
        return item_class.from_list(item)


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
        # now we want nice class methods for serializing

        @classmethod
        def from_dict(ncls, item_dict):
            params = inspect.signature(ncls.__init__).parameters
            args = []
            for name, param in params.items():
                if name == "self":
                    continue
                if name not in item_dict:
                    if name not in getattr(ncls, '_glotaran_attribute_defaults'):
                        raise Exception(f"Missing parameter '{name} for item "
                                        f"'{ncls.__name__}'")
                    args.append(getattr(ncls, '_glotaran_attribute_defaults')[name])
                else:
                    item = item_dict[name]
                    item_class = param.annotation
                    args.append(item_or_list_to_arg(name, item, item_class))

            return ncls(*args)

        setattr(cls, 'from_dict', from_dict)

        @classmethod
        def from_list(ncls, item_list):
            names = [n for n in
                     inspect.signature(ncls.__init__).parameters if not n == "self"]
            params = [p for _, p in
                      inspect.signature(ncls.__init__).parameters.items()]
            # params contains 'self'
            params = params[1:]
            if len(item_list) is not len(params):
                raise Exception(f"To few or much parameters for '{ncls.__name__}'"
                                f"\nGot: {item_list}\nWant: {names}")

            for i in range(len(item_list)):
                item_class = params[i].annotation
                item_list[i] = item_or_list_to_arg(names[i], item_list[i], item_class)

            return ncls(*item_list)

        setattr(cls, 'from_list', from_list)

        validator = Validator(cls)

        def val_model(self, model, errors=[], validator=validator):
            return validator.val_model(self, model, errors)

        setattr(cls, 'validate_model', val_model)

        def val_parameter(self, model, parameter, errors=[],
                          validator=validator):
            return validator.val_parameter(self, model, parameter, errors)
        setattr(cls, 'validate_parameter', val_parameter)

        def fill(self, model, parameter):

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

        setattr(cls, 'fill', fill)

        mprint = _create_mprint_func(cls)
        setattr(cls, 'mprint', mprint)
        #  setattr(cls, '__str__', functools.wraps(cls.__str__)(mprint))

        return cls

    return decorator


def model_item_typed(types: Dict[str, any] = {}, no_label=False):
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


def _create_mprint_func(cls):

    def mprint_item(self, parameter: ParameterGroup = None, initial: ParameterGroup = None):
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

    mprint_item.__annotations__ = {
        'return': str,
    }
    mprint_item.__name__ = 'mprint'
    mprint_item.__qualname__ = cls.__qualname__ + '.' + mprint_item.__name__
    mprint_item.__module__ = cls.__module__
    mprint_item.__doc__ = f'''
    Returns a string with the item formatted in markdown.
    '''

    return mprint_item
