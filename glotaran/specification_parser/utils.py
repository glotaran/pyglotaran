from itertools import chain


def get_keys_from_object(obj, keys, start=0):
    if is_compact(obj):
        idx = [i+start for i in range(len(keys))]
        return tuple(list([obj[i] if len(obj) > i else None for i in idx]))
    else:
        return tuple(list([obj[k] if k in obj else None for k in keys]))


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


def parse_range(rng):
    parts = rng.split('-')

    if 1 > len(parts) > 2:
        raise ValueError("Bad range: '%s'" % (rng,))
    start = parts[0]
    if len(parts) == 1:
        yield "{}".format(start).strip()
    else:
        path = []
        if not start.isdigit():
            path = start.split(".")
            start = path.pop()
        start = int(start)
        end = parts[1]
        if not end.isdigit():
            end = end.split(".").pop()
        end = int(end)
        if start > end:
            end, start = start, end
        if path:
            for i in range(start, end + 1):
                yield ".".join(path) + ".{}".format(i).strip()
        else:
            for i in range(start, end + 1):
                yield "{}".format(i).strip()


def parse_range_list(rngs):
    return sorted(set(chain(*[parse_range(rng) for rng in rngs.split(',')])))


def is_compact(obj):
    return isinstance(obj, list)
