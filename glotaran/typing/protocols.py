"""Protocol like type definitions."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from glotaran.typing.types import StrOrPath


class FileLoadableProtocol(Protocol):
    """Protocol class that a file loadable class need adherer to."""

    loader: Callable[
        [StrOrPath | Sequence[StrOrPath] | Mapping[str, StrOrPath]],
        FileLoadableProtocol,
    ]
    source_path: StrOrPath | Sequence[StrOrPath] | Mapping[str, StrOrPath]


FileLoadable = TypeVar("FileLoadable", bound=FileLoadableProtocol)
