"""Protocol like type definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

if TYPE_CHECKING:
    from glotaran.typing.types import Self
    from glotaran.typing.types import StrOrPath


class ToFromCsvSerializable(Protocol):
    """Protocol class that a CSV serializable class needs to adhere to."""

    def to_csv(self, path: StrOrPath, delimiter: str = ",") -> None: ...

    @classmethod
    def from_csv(cls: type[Self], path: StrOrPath) -> Self: ...
