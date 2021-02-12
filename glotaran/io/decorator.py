from __future__ import annotations

from glotaran.register.register import register_io


def io(fmt: str | list(str)):
    def decorator(cls):
        register_io(fmt, cls)
        return cls

    return decorator


def implements(method):
    setattr(method, "implements", True)
    return method
