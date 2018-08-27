from typing import List
from dataclasses import dataclass
import inspect


class MissingParameterException(Exception):
    pass


def item_or_list_to_param(name, item, param_class):
    islist = issubclass(param_class, List)
    if islist:
        param_class = param_class.__args__[0]
    if not hasattr(param_class, "_glotaran_model_item"):
        return item
    if not islist:
        return item_to_param(name, item, param_class)
    for i in range(len(item)):
        item[i] = item_to_param(name, item[i], param_class)
    return item

def item_to_param(name, item, param_class):
    is_typed = hasattr(param_class, "_glotaran_model_item_typed")
    if isinstance(item, dict):
        if is_typed:
            if 'type' not in item:
                raise Exception(f"Missing type for attribute '{name}'")
            item_type = item['type']

            if item_type not in param_class._glotaran_model_item_types:
                raise Exception(f"Unknown type '{item_type}' "
                                f"for attribute '{name}'")
            param_class = \
                param_class._glotaran_model_item_types[item_type]
        return param_class.from_dict(item)
    else:
        if is_typed:
            print(item)
            if len(item) < 2 and len(item) is not 1:
                raise Exception(f"Missing type for attribute '{name}'")
            item_type = item[1] if len(item) is not 1 else item[0]

            if item_type not in param_class._glotaran_model_item_types:
                raise Exception(f"Unknown type '{item_type}' "
                                f"for attribute '{name}'")
            param_class = \
                param_class._glotaran_model_item_types[item_type]
        return param_class.from_list(item)


def glotaran_model_item(attributes={},
                        parameter=[],
                        has_type=False,
                        no_label=False):
    def decorator(cls):

        # Set annotations for autocomplete and doc

        if not no_label:
            annotations = {'label': str}
        if has_type:
            annotations = {'type': str}
        for name, aclass in attributes.items():
            if isinstance(aclass, tuple):
                (aclass, default) = aclass
                setattr(cls, name, default)
            annotations[name] = aclass
        setattr(cls, '__annotations__', annotations)

        # We turn it into dataclass to get automagic inits
        cls = dataclass(cls)

        # for nesting
        setattr(cls, '_glotaran_model_item', True)

        # store the fit parameter for later sanity checking
        setattr(cls, '_glotaran_fit_parameter', parameter)

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
        return cls

    return decorator


def glotaran_model_item_typed(types={}):

    def decorator(cls):

        setattr(cls, '_glotaran_model_item', True)
        setattr(cls, '_glotaran_model_item_typed', True)
        setattr(cls, '_glotaran_model_item_types', types)

        return cls

    return decorator
