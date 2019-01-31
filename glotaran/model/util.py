import typing

from glotaran.parameter import Parameter


def wrap_func_as_method(cls, name=None, annotations=None, doc=None):

    def decorator(func):

        if name:
            func.__name__ = name
        if annotations:
            func.__annotations__ = annotations
        if doc:
            func.__doc__ = doc
        func.__qualname__ = cls.__qualname__ + '.' + func.__name__
        func.__module__ = cls.__module__

        return func

    return decorator


def is_list_type(item_class):
    return issubclass(item_class.__origin__, typing.List) if \
        hasattr(item_class, '__origin__') else False


def is_dict(item_class):
    return issubclass(item_class.__origin__, typing.Dict) if \
        hasattr(item_class, '__origin__') else False


def is_item_or_list_of(item_class):
    if not isinstance(item_class, type):
        return False
    islist = is_list_type(item_class)
    if islist:
        item_class = item_class.__args__[0]
    return hasattr(item_class, "_glotaran_model_item")


def item_or_list_to_arg(name, item, item_class):
    islist = is_list_type(item_class)
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
    elif is_dict(item_class):
        v_class = item_class.__args__[1]
        islist = is_list_type(v_class)
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
