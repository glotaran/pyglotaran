"""The glotaran registry module."""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from glotaran.utils.ipython import MarkdownStr


class ProjectRegistry:
    """A registry base class."""

    def __init__(
        self, directory: Path, file_suffix: str | Iterable[str], loader: Callable, item_name: str
    ):
        """Initialize a registry.

        Parameters
        ----------
        directory : Path
            The registry directory.
        file_suffix : str | Iterable[str]
            The suffixes of item files.
        loader : Callable
            A loader for the registry items.
        """
        self._directory: Path = directory
        self._file_suffix: tuple[str, ...] = (
            (file_suffix,) if isinstance(file_suffix, str) else tuple(file_suffix)
        )
        self._loader: Callable = loader
        self._item_name = item_name

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
        dict[str, Path]
            The items of the registry.
        """
        items = {path.stem: path for path in self._directory.iterdir() if self.is_item(path)}
        return dict(sorted(items.items()))

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
        if name in self.items:
            return self._loader(self.items[name])
        raise ValueError(
            f"{self._item_name} '{name}' does not exist. "
            f"Known {self._item_name.rstrip('s')}s are: {list(self.items.keys())}"
        )

    def markdown(self, join_indentation: int = 0) -> MarkdownStr:
        """Format the registry items as a markdown text.

        Parameters
        ----------
        join_indentation: int
            Number of whitespaces to indent when joining the parts.
            This is intended to be used with dedent when used in an indented f-string.
            Defaults to 0.

        Returns
        -------
        MarkdownStr : str
            The markdown string.
        """
        if self.empty:
            return MarkdownStr("_None_")

        join_str = " " * join_indentation
        md = join_str.join(f"* {name}\n" for name in self.items)
        return MarkdownStr(md)

    def _create_directory_if_not_exist(self):
        """Create the registry directory if it does not exist."""
        self._directory.mkdir(parents=True, exist_ok=True)
