"""The glotaran model registry module."""
from __future__ import annotations

from pathlib import Path

from glotaran.io import load_model
from glotaran.plugin_system.project_io_registration import supported_file_extensions_project_io
from glotaran.project.generators.generator import GeneratorArguments
from glotaran.project.generators.generator import generate_model_yml
from glotaran.project.project_registry import ProjectRegistry


class ProjectModelRegistry(ProjectRegistry):
    """A registry for models."""

    def __init__(self, directory: Path):
        """Initialize a model registry.

        Parameters
        ----------
        directory : Path
            The registry directory.
        """
        super().__init__(
            directory / "models",
            supported_file_extensions_project_io("load_model"),
            load_model,
            item_name="Model",
        )

    def generate_model(
        self,
        name: str,
        generator_name: str,
        generator_arguments: GeneratorArguments,
        *,
        allow_overwrite: bool = False,
        ignore_existing: bool = False,
    ):
        """Generate a model.

        Parameters
        ----------
        name : str
            The name of the model.
        generator_name : str
            The generator for the model.
        generator_arguments : GeneratorArguments
            Arguments for the generator.
        allow_overwrite: bool
            Whether to overwrite an existing model.
        ignore_existing: bool
            Whether to ignore generation of a model file if it already exists.

        Raises
        ------
        FileExistsError
            Raised if model is already existing and `allow_overwrite=False`.
        """
        model_yml = generate_model_yml(
            generator_name=generator_name, generator_arguments=generator_arguments
        )
        model_file = self._directory / f"{name}.yml"
        if model_file.exists() and ignore_existing:
            return

        if model_file.exists() and not allow_overwrite:
            raise FileExistsError(f"Model {name!r} already exists and `allow_overwrite=False`.")

        model_file.write_text(model_yml)
