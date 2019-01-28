
import typing
import functools
import holoviews as hv


def saveable(func: typing.Callable) -> typing.Callable:

    @functools.wraps(func)
    def decorated(*args, filename=None, **kwargs):
        curve = func(*args, **kwargs)
        if filename:
            hv.save(curve, filename)
        return curve

    return decorated
