from typing import List
from dataclasses import dataclass, replace
import inspect


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
            item_type = item[1] if len(item) is not 1 else item[0]

            if item_type not in item_class._glotaran_model_item_types:
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

        # Set annotations for autocomplete and doc
        validate_nested = []
        annotations = {}
        if not no_label:
            annotations['label'] = str
        if has_type:
            annotations['type'] = str
        for name, item_class in attributes.items():
            if isinstance(item_class, tuple):
                (item_class, default) = item_class
                setattr(cls, name, default)
            annotations[name] = item_class
            if is_item_or_list_of(item_class):
                validate_nested.append(name)
        setattr(cls, '__annotations__', annotations)

        # We turn it into dataclass to get automagic inits
        cls = dataclass(cls)

        # for nesting
        setattr(cls, '_glotaran_model_item', True)

        # store for later sanity checking
        setattr(cls, '_glotaran_attributes', [name for name in attributes])

        # strore the number of attributes
        setattr(cls, '_glotaran_nr_attributes', len(attributes))

        # now we want nice class methods for serializing

        @classmethod
        def from_dict(ncls, item_dict):
            params = inspect.signature(ncls.__init__).parameters
            args = []
            for name, param in params.items():
                if name == "self":
                    continue
                if name not in item_dict:
                    if param.default is param.empty:
                        raise MissingParameterException(name)
                    args.append(param.default)
                else:
                    item = item_dict[name]
                    item_class = param.annotation
                    args.append(item_or_list_to_param(name, item, item_class))

            return ncls(*args)

        setattr(cls, 'from_dict', from_dict)

        @classmethod
        def from_list(ncls, item_list):
            names = [n for n in
                     inspect.signature(ncls.__init__).parameters]
            params = [p for _, p in
                      inspect.signature(ncls.__init__).parameters.items()]
            # params contains 'self'
            if len(item_list) is not len(params)-1:
                raise MissingParameterException

            for i in range(len(item_list)):
                item_class = params[i].annotation
                item_list[i] = item_or_list_to_param(names[i], item_list[i], item_class)

            return ncls(*item_list)

        setattr(cls, 'from_list', from_list)

        def val_model(self, model, errors=[]):
            attrs = getattr(self, '_glotaran_attributes')
            for attr in attrs:
                item = getattr(self, attr)
                if hasattr(model, attr):
                    labels = item
                    if not isinstance(labels, list):
                        labels = [labels]
                    model_attr = getattr(model, attr)
                    for label in labels:
                        if label not in model_attr:
                            errors.append(f"Missing '{attr}' with label '{label}'")

                else:
                    nested = item
                    if not isinstance(nested, list):
                        nested = [nested]
                    for n in nested:
                        if hasattr(n, "_glotaran_model_item"):
                            n.validate_model(model, errors=errors)

            return errors

        setattr(cls, 'validate_model', val_model)

        def val_parameter(self, model, parameter, errors=[]):
            attrs = getattr(self, '_glotaran_attributes')
            for attr in attrs:
                item = getattr(self, attr)
                if not hasattr(model, attr):
                    if not isinstance(item, list):
                        item = [item]
                    if any([not isinstance(i, str) for i in item]):
                        continue
                    for label in item:
                        if not parameter.has(label):
                            errors.append(f"Missing parameter with label '{label}'")

                else:
                    nested = item
                    if not isinstance(nested, list):
                        nested = [nested]
                    for n in nested:
                        if hasattr(n, "_glotaran_model_item"):
                            n.validate_parameter(model, parameter, errors=errors)

            return errors

        setattr(cls, 'validate_parameter', val_parameter)

        def fill(self, model, parameter):

            def convert(item):
                if not isinstance(item, str):
                    return item
                if hasattr(item, "_glotaran_model_item"):
                    return item.fill(model, parameter)
                return parameter.get(item).value

            replaced = {}
            attrs = getattr(self, '_glotaran_attributes')
            for attr in attrs:
                item = getattr(self, attr)
                if hasattr(model, attr):
                    model_attr = getattr(model, attr)
                    if isinstance(item, list):
                        item = [model_attr[i].fill(model, parameter) for i in item]
                    else:
                        item = model_attr[item].fill(model, parameter)
                elif isinstance(item, list):
                    item = [convert(i) for i in item]
                else:
                    item = convert(item)
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
