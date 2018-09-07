from typing import List
from dataclasses import dataclass, replace
import inspect

from .validator import Validator

class MissingParameterException(Exception):
    pass


def is_item_or_list_of(item_class):
    if not isinstance(item_class, type):
        return False
    islist = issubclass(item_class, List)
    if islist:
        item_class = item_class.__args__[0]
    return hasattr(item_class, "_glotaran_model_item")


def item_or_list_to_param(name, item, item_class):
    islist = issubclass(item_class, List)
    if islist:
        item_class = item_class.__args__[0]
    if not hasattr(item_class, "_glotaran_model_item"):
        return item
    if not islist:
        return item_to_param(name, item, item_class)
    for i in range(len(item)):
        print(item)
        item[i] = item_to_param(name, item[i], item_class)
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
                    hasattr(item_class,'label') else item[0]

            if item_type not in item_class._glotaran_model_item_types:
                print(item)
                raise Exception(f"Unknown type '{item_type}' "
                                f"for attribute '{name}'")
            item_class = \
                item_class._glotaran_model_item_types[item_type]
        return item_class.from_list(item)


def glotaran_model_item(attributes={},
                        has_type=False,
                        no_label=False):
    """glotaran_model_item adds the given attributes to the class and applies
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

        # Set annotations for autocomplete and doc
        validate_nested = []
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
            if is_item_or_list_of(item_class):
                validate_nested.append(name)

        setattr(cls, '__annotations__', annotations)

        # We turn it into dataclass to get automagic inits
        cls = dataclass(cls)

        # for nesting
        setattr(cls, '_glotaran_model_item', True)

        # store for later sanity checking
        if not hasattr(cls, '_glotaran_attributes'):
            setattr(cls, '_glotaran_attributes', {})
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
                        raise MissingParameterException(f"Missing parameter '{name} for item "
                                                        f"'{ncls.__name__}'")
                    args.append(getattr(ncls, '_glotaran_attribute_defaults')[name])
                else:
                    item = item_dict[name]
                    item_class = param.annotation
                    args.append(item_or_list_to_param(name, item, item_class))

            return ncls(*args)

        setattr(cls, 'from_dict', from_dict)

        @classmethod
        def from_list(ncls, item_list):
            names = [n for n in
                     inspect.signature(ncls.__init__).parameters if not n == "self"]
            params = [p for _, p in
                      inspect.signature(ncls.__init__).parameters.items()]
            # params contains 'self'
            if len(item_list) is not len(params)-1:
                raise MissingParameterException(f"To few or much parameters for '{ncls.__name__}'\n"
                                                f" Got: {item_list}\nWant: {names}")

            for i in range(len(item_list)):
                item_class = params[i].annotation
                item_list[i] = item_or_list_to_param(names[i], item_list[i], item_class)

            return ncls(*item_list)

        setattr(cls, 'from_list', from_list)

        validator = Validator(attributes)

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
                    return cp
                if not isinstance(item, str):
                    return item
                if hasattr(item, "_glotaran_model_item"):
                    return item.fill(model, parameter)
                return parameter.get(item).value

            def fill_item_or_list(item, attr):
                model_attr = getattr(model, attr)
                if isinstance(item, list):
                    return [model_attr[i].fill(model, parameter) for i in item]
                return model_attr[item].fill(model, parameter)

            replaced = {}
            attrs = getattr(self, '_glotaran_attributes')
            print("attrs", attrs)
            for attr, opts in attrs.items():
                item = getattr(self, attr)
                print("attr", attr, opts)
                print(item)
                if item is None:
                    continue
                if 'target' in opts:
                    target = opts['target'][1]
                    print('target', target)
                    nitem = {}
                    for k, v in item.items():
                        if target == 'parameter':
                            nitem[k] = convert_list_or_scalar(v)
                        else:
                            nitem[k] = fill_item_or_list(v, target)
                    item = nitem

                elif hasattr(model, attr):
                    item = fill_item_or_list(item, attr)
                else:
                    item = convert_list_or_scalar(item)
                replaced[attr] = item
            return replace(self, **replaced)

        setattr(cls, 'fill', fill)


        return cls

    return decorator


def glotaran_model_item_typed(types={}):

    def decorator(cls):

        setattr(cls, '_glotaran_model_item', True)
        setattr(cls, '_glotaran_model_item_typed', True)
        setattr(cls, '_glotaran_model_item_types', types)

        return cls

    return decorator
