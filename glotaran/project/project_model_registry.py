"""The glotaran model registry module."""
from __future__ import annotations

from pathlib import Path

from glotaran.io import load_model
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
        super().__init__(directory / "models", ".yml", load_model)

    def generate_model(
        self,
        name: str,
        generator_name: str,
        generator_arguments: GeneratorArguments,
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
        """
        model_yml = generate_model_yml(
            generator_name=generator_name, generator_arguments=generator_arguments
        )
        (self._directory / f"{name}.yml").write_text(model_yml)
