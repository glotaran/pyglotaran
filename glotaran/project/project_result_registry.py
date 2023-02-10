"""The glotaran result registry module."""
from __future__ import annotations

import re
from pathlib import Path
from warnings import warn

from glotaran.io import load_result
from glotaran.io import save_result
from glotaran.project.project_registry import ProjectRegistry
from glotaran.project.result import Result


class ProjectResultRegistry(ProjectRegistry):
    """A registry for results."""

    result_pattern = re.compile(r".+_run_\d{4}$")

    def __init__(self, directory: Path):
        """Initialize a result registry.

        Parameters
        ----------
        directory : Path
            The registry directory.
        """
        super().__init__(
            directory / "results",
            [],
            lambda path: load_result(path / "result.yml", format_name="yml"),
            item_name="Result",
        )

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
        return path.is_dir()

    def previous_result_paths(self, base_name: str) -> list[Path]:
        """List previous result paths with base_name.

        Parameters
        ----------
        base_name: str
            The base name for the result provided by user or derived from model name.

        Returns
        -------
        list[Path]
            Paths to previous results with name ``base_name``.
        """
        return sorted(self.directory.glob(f"{base_name}_run_*"))

    def _latest_result_path_fallback(self, name: str, *, latest: bool = False) -> Path:
        """Fallback when a user forgets to specify the run to get a result.

        If ``name`` contains the run number this will just return ``name``,
        else we try to get the name of the latest run.

        Parameters
        ----------
        name: str
            Name of the result, which should contain the run specifier.
        latest: bool
            Flag to deactivate warning about using latest result. Defaults to False.


        Returns
        -------
        Path
            Path to the result (latest result if ``name`` does not match the result pattern).

        Raises
        ------
        ValueError
            Raised if result does not exist.
        """
        if re.match(self.result_pattern, name) is None:
            if latest is False:
                warn(
                    UserWarning(
                        f"Result name {name!r} is missing the run specifier, "
                        "falling back to try getting latest result. "
                        "Use latest=True to mute this warning."
                    ),
                    stacklevel=3,
                )
            previous_result_paths = self.previous_result_paths(name) or [Path(name)]
            name = previous_result_paths[-1].stem
        path = self._directory / name
        if self.is_item(path):
            return path

        raise ValueError(
            f"Result {name!r} does not exist. Known Results are: {list(self.items.keys())}"
        )

    def create_result_run_name(self, base_name: str) -> str:
        """Create a result name for a model.

        Parameters
        ----------
        base_name: str
            The base name for the result provided by user or derived from model name.

        Returns
        -------
        str :
            Folder name for the new result to be saved in.
        """
        previous_results = self.previous_result_paths(base_name)
        if not previous_results:
            return f"{base_name}_run_0000"
        latest_result_run_nr = int(previous_results[-1].stem.replace(f"{base_name}_run_", ""))
        return f"{base_name}_run_{latest_result_run_nr+1:04}"

    def save(self, name: str, result: Result):
        """Save a result.

        Parameters
        ----------
        name : str
            The name of the result.
        result : Result
            The result to save.
        """
        run_name = self.create_result_run_name(name)
        result_path = self.directory / run_name / "result.yml"
        save_result(result, result_path, format_name="yml")
