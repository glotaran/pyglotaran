"""The glotaran model registry module."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from glotaran.io import load_model
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
        generator: str,
        generator_arguments: dict[str, Any],
    ):
        """Generate a model.

        Parameters
        ----------
        name : str
            The name of the model.
        generator : str
            The generator for the model.
        generator_arguments : dict[str, Any]
            Arguments for the generator.
        """
        model_yml = generate_model_yml(generator, **generator_arguments)
        with open(self._directory / f"{name}.yml", "w") as f:
            f.write(model_yml)
