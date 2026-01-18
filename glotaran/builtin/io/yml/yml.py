"""Module containing the YAML Data and Project IO plugins."""

from __future__ import annotations

import os
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.io.interface import SAVING_OPTIONS_DEFAULT
from glotaran.io.interface import SavingOptions
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.project.result import Result
from glotaran.utils.sanitize import sanitize_yaml

if TYPE_CHECKING:
    from typing import Any


@register_project_io(["yml", "yaml", "yml_str"])
class YmlProjectIo(ProjectIoInterface):
    """Plugin for YAML project io."""

    def load_parameters(self, file_name: str) -> Parameters:
        """Load :class:`Parameters` instance from the specification defined in ``file_name``.

        Parameters
        ----------
        file_name: str
            File containing the parameter specification.

        Returns
        -------
        ``Parameters``
        """  # noqa:  D414
        spec = self._load_yml(file_name)

        if isinstance(spec, list):
            return Parameters.from_list(spec)
        return Parameters.from_dict(spec)

    def load_scheme(self, file_name: str) -> Scheme:
        """Load :class:`Scheme` instance from the specification defined in ``file_name``.

        Parameters
        ----------
        file_name: str
            File containing the scheme specification.

        Returns
        -------
        Scheme
        """
        spec = sanitize_yaml(self._load_yml(file_name), do_values=True)
        return Scheme.from_dict(
            spec,
            source_path=Path(file_name) if os.path.isfile(file_name) else None,  # noqa: PTH113
        )

    def save_scheme(self, scheme: Scheme, file_name: str) -> None:
        """Write a :class:`Scheme` instance to a specification file ``file_name``.

        Parameters
        ----------
        scheme: Scheme
            :class:`Scheme` instance to save to file.
        file_name: str
            Path to the file to write the scheme specification to.
        """
        scheme_file = Path(file_name)
        if (
            scheme.source_path is not None
            and scheme.source_path.suffix in (".yml", ".yaml")
            and self.load_scheme(str(scheme.source_path)).model_dump(exclude_unset=True)
            == scheme.model_dump(exclude_unset=True)
        ):
            if scheme.source_path == scheme_file:
                return
            copyfile(scheme.source_path, file_name)
        else:
            write_dict(scheme.model_dump(exclude_unset=True, mode="json"), file_name=file_name)

    def load_result(self, result_path: str) -> Result:
        """Create a :class:`Result` instance from the specs defined in a file.

        Parameters
        ----------
        result_path : str
            Path containing the result data.

        Returns
        -------
        Result
            :class:`Result` instance created from the saved format.
        """
        result_file_path = Path(result_path)
        if result_file_path.suffix not in [".yml", ".yaml"]:
            result_file_path = result_file_path / "result.yml"
        spec = self._load_yml(result_file_path.as_posix())
        return Result.model_validate(spec, context={"save_folder": result_file_path.parent})

    def save_result(
        self,
        result: Result,
        result_path: str,
        saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT,
    ) -> list[str]:
        """Write a :class:`Result` instance to a specification file and data files.

        Returns a list with paths of all saved items.
        The following files are saved if not configured otherwise:
        * ``result.yml``: Yaml spec file of the result
        * ``scheme.yml``: Scheme spec file.
        * ``initial_parameters.csv``: Initially used parameters.
        * ``optimized_parameters.csv``: The optimized parameter as csv file.
        * ``parameter_history.csv``: Parameter changes over the optimization
        * ``optimization_history.csv``: Parsed table printed by the SciPy optimizer
        * ``optimization_results/**/{dataset_label}.nc``: The results as NetCDF files.

        Parameters
        ----------
        result: Result
            :class:`Result` instance to write.
        result_path: str
            Path to write the result data to.
        saving_options: SavingOptions
            Options for saving the the result.

        Returns
        -------
        list[str]
            List of file paths which were created.
        """
        result_file_path = Path(result_path)
        if result_file_path.suffix not in [".yml", ".yaml"]:
            result_file_path = result_file_path / "result.yml"
        save_folder = result_file_path.parent
        spec = result.model_dump(
            exclude_unset=True,
            exclude_defaults=True,
            mode="json",
            context={"save_folder": save_folder, "saving_options": saving_options},
        )
        write_dict(spec, file_name=result_file_path.as_posix())

        return Result.extract_paths_from_serialization(result_file_path, spec)

    def _load_yml(self, file_name: str) -> dict[str, Any]:
        return load_dict(file_name, is_file=self.format != "yml_str")
