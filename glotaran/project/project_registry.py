"""The glotaran registry module."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Callable

from glotaran.utils.ipython import MarkdownStr


class ProjectRegistry:
    """A registry base class."""

    def __init__(self, directory: Path, file_suffix: str | list[str], loader: Callable):
        """Initialize a registry.

        Parameters
        ----------
        directory : Path
            The registry directory.
        file_suffix : str | list[str]
            The suffixes of item files.
        loader : Callable
            A loader for the registry items.
        """
        self._directory: Path = directory
        self._file_suffix: list[str] = (
            [file_suffix] if isinstance(file_suffix, str) else file_suffix
        )
        self._loader: Callable = loader

        self._create_directory_if_not_exist()

    @property
    def directory(self) -> Path:
        """Get the registry directory.

        Returns
        -------
        Path
            The registry directory.
        """
        return self._directory

    @property
    def empty(self) -> bool:
        """Whether the registry is empty.

        Returns
        -------
        bool
            Whether the registry is empty.
        """
        return len(self.items) == 0

    @property
    def items(self) -> dict[str, Path]:
        """Get the items of the registry.

        Returns
        -------
        dict[str, str]
            The items of the registry.
        """
        return {
            path.with_suffix("").name: path
            for path in self._directory.iterdir()
            if self.is_item(path)
        }

    def is_item(self, path: Path) -> bool:
        """Check if the path contains an registry item.

        Parameters
        ----------
        path : Path
            The path to check.

        Returns
        -------
        bool :
            Whether the path contains an item.
        """
        return path.suffix in self._file_suffix

    def load_item(self, name: str) -> Any:
        """Load an registry item by it's name.

        Parameters
        ----------
        name : str
            The item name.

        Returns
        -------
        Any
            The loaded item.

        Raises
        ------
        ValueError
            Raise if the item does not exist.
        """
        try:
            path = next(p for p in self._directory.iterdir() if name in p.name)
        except StopIteration:
            raise ValueError(f"No Item with name '{name}' exists.")
        return self._loader(path)

    def markdown(self) -> MarkdownStr:
        """Format the registry items as a markdown text.

        Returns
        -------
        MarkdownStr : str
            The markdown string.
        """
        if self.empty:
            return MarkdownStr("_None_")

        md = ""
        for name in self.items:
            md += f"* {name}\n"

        return MarkdownStr(md)

    def _create_directory_if_not_exist(self):
        """Create the registry directory if it does not exist."""
        if not self._directory.exists():
            self._directory.mkdir()