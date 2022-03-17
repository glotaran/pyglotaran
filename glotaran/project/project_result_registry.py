"""The glotaran result registry module."""
from __future__ import annotations

from pathlib import Path

from glotaran.io import load_result
from glotaran.io import save_result
from glotaran.project.project_registry import ProjectRegistry
from glotaran.project.result import Result


class ProjectResultRegistry(ProjectRegistry):
    """A registry for results."""

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

    def create_result_name_for_model(self, model_name: str) -> str:
        """Create a result name for a model.

        Parameters
        ----------
        model_name : str
            The model name.

        Returns
        -------
        str :
            A result name.
        """
        i = 0
        while True:
            result_name = f"{model_name}_run_{i}"
            if not (self.directory / result_name).exists():
                return result_name
            i += 1

    def save(self, name: str, result: Result):
        """Save a result.

        Parameters
        ----------
        name : str
            The name of the result.
        result : Result
            The result to save.
        """
        result_path = self.directory / name / "result.yml"
        save_result(result, result_path, format_name="yml", allow_overwrite=True)