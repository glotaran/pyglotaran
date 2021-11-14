from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar

if TYPE_CHECKING:
    from typing import Callable

    from glotaran.typing import StrOrPath


class FileLoadableProtocol(Protocol):
    loader: Callable[[StrOrPath], FileLoadableProtocol]
    source_path: StrOrPath


FileLoadable = TypeVar("FileLoadable", bound=FileLoadableProtocol)
