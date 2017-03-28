def get_keys_from_object(obj, keys, start=0):
    if is_compact(obj):
        keys = [i+start for i in range(len(keys))]
    return (obj[k] for k in keys)


def retrieve_optional(obj, key, index, default):
    compact = is_compact(obj)
    if compact:
        if len(obj) <= index:
            return default
        return obj[index]
    else:
        if key in obj:
            return obj[key]
        return default


def is_compact(obj):
    return isinstance(obj, list)
