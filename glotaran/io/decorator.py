from __future__ import annotations

from .register import register_io as register


def register_io(fmt: str | list(str)):
    def decorator(cls):
        register(fmt, cls)
        return cls

    return decorator
