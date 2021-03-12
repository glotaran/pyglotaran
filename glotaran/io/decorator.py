from __future__ import annotations

from typing import TYPE_CHECKING

from .register import _register_project_io

if TYPE_CHECKING:
    from glotaran.io.register import ProjectIoInterface


def register_project_io(fmt: str | list[str]):
    def decorator(cls: type[ProjectIoInterface]):
        _register_project_io(fmt, cls)
        return cls

    return decorator
