from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar

if TYPE_CHECKING:
    from os import PathLike
    from typing import Callable


class FileLoadableProtocol(Protocol):
    loader: Callable[[str | PathLike[str]], FileLoadableProtocol]
    source_path: str | PathLike[str]


FileLoadable = TypeVar("FileLoadable", bound=FileLoadableProtocol)
