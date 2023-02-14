"""The glotaran registry module."""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from warnings import warn

from glotaran.utils.ipython import MarkdownStr


class AmbiguousNameWarning(UserWarning):
    """Warning thrown when an item with the same name already exists.

    This is the case if two files with the same name but different extensions exist next to
    each other.
    """

    def __init__(
        self,
        *,
        item_name: str,
        items: dict[str, Path],
        item_key: str,
        unique_item_key: str,
        project_root_path: Path,
    ):
        """Initialize ``AmbiguousNameWarning`` with a formatted message.

        Parameters
        ----------
        item_name: str
            Name of items in the registry (e.g. 'Parameters').
        items: dict[str, Path]
            Known items at this iteration point.
        item_key: str
            Key that would have been used if the file names weren't ambiguous.
        unique_item_key: str
            Unique key for the item with ambiguous file name.
        project_root_path: Path
            Root path of the project.
        """
        ambiguous_file_names = [
            value.relative_to(project_root_path).as_posix()
            for key, value in items.items()
            if key.startswith(item_key)
        ]
        super().__init__(
            f"The {item_name} name {item_key!r} is ambiguous since it could "
            f"refer to the following files: {ambiguous_file_names}\n"
            f"The file {items[unique_item_key].relative_to(project_root_path).as_posix()!r} "
            f"will be accessible by the name {unique_item_key!r}. \n"
            f"While {item_key!r} refers to the file "
            f"{items[item_key].relative_to(project_root_path).as_posix()!r}.\n"
            "Rename the files with unambiguous names to silence this warning."
        )


class ItemMapping(Mapping):
    """Container class for ``ProjectRegistry`` items.

    The main purpose of this class is to show a user friendly error when accessing none existing
    items.
    """

    def __init__(self, data: Mapping[str, Path], item_name: str) -> None:
        """Initialize class instance as wrapper around ``data``.

        Parameters
        ----------
        data: Mapping[str, Path]
            Underlying data that are used for mapping.
        item_name: str
            Name of items in the registry used to format warning and exception (e.g. 'Parameters').
        """
        self.data = dict(sorted(data.items()))
        self._item_name = item_name

    def __getitem__(self, key: str) -> Path:
        """Protocol method used when accessing an item."""
        if key in self.data:
            return self.data[key]
        raise ValueError(
            f"{self._item_name} '{key}' does not exist. "
            f"Known {self._item_name.rstrip('s')}s are: {list(self.data.keys())}"
        )

    def __iter__(self) -> Iterator[str]:
        """Protocol method used when iterating over an instance."""
        return iter(self.data)

    def __len__(self) -> int:
        """Protocol method used by ``len``."""
        return len(self.data)

    def __eq__(self, other: object) -> bool:
        """Protocol method used for equality checks."""
        if isinstance(other, ItemMapping):
            return self.data == other.data
        if isinstance(other, Mapping):  # sourcery skip: assign-if-exp
            return self.data == other
        return NotImplemented

    def __repr__(self) -> str:
        """Protocol method used to display instance."""
        return repr(self.data)


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
        item_name: str
            Name of items in the registry used to format warning and exception (e.g. 'Parameters').
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
    def items(self) -> ItemMapping:
        """Get the items of the registry.

        Returns
        -------
        ItemMapping
            The items of the registry.
        """
        items = {}
        for path in sorted(self._directory.rglob("*")):
            if self.is_item(path) is True:
                rel_parent_path = path.parent.relative_to(self._directory)
                item_key = (rel_parent_path / path.stem).as_posix()
                if item_key not in items:
                    items[item_key] = path
                else:
                    unique_item_key = (rel_parent_path / path.name).as_posix()
                    items[unique_item_key] = path
                    warn(
                        AmbiguousNameWarning(
                            item_name=self._item_name,
                            items=items,
                            item_key=item_key,
                            unique_item_key=unique_item_key,
                            project_root_path=self._directory.parent,
                        ),
                        stacklevel=3,
                    )
        return ItemMapping(items, self._item_name)

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


        .. # noqa: DAR402
        """
        return self._loader(self.items[name])

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
