"""Module containing the YAML Data and Project IO plugins."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.builtin.io.yml.utils import write_dict
from glotaran.deprecation.modules.builtin_io_yml import model_spec_deprecations
from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.io import ProjectIoInterface
from glotaran.io import SavingOptions
from glotaran.io import register_project_io
from glotaran.io import save_model
from glotaran.io import save_result
from glotaran.io import save_scheme
from glotaran.model import Model
from glotaran.parameter import Parameters
from glotaran.plugin_system.megacomplex_registration import get_megacomplex
from glotaran.project.dataclass_helpers import asdict
from glotaran.project.dataclass_helpers import fromdict
from glotaran.project.project import Result
from glotaran.project.scheme import Scheme
from glotaran.utils.sanitize import sanitize_yaml

if TYPE_CHECKING:
    from typing import Any


@register_project_io(["yml", "yaml", "yml_str"])
class YmlProjectIo(ProjectIoInterface):
    """Plugin for YAML project io."""

    def load_model(self, file_name: str) -> Model:
        """Load a :class:`Model` from a model specification in a yaml file.

        Parameters
        ----------
        file_name: str
            Path to the model file to read.

        Raises
        ------
        ValueError
            If ``megacomplex`` was not provided in the model specification.
        ValueError
            If ``default_megacomplex`` was not provided and any megacomplex is missing the type
            attribute.

        Returns
        -------
        Model
        """
        spec = self._load_yml(file_name)

        model_spec_deprecations(spec)

        spec = sanitize_yaml(spec)

        if "megacomplex" not in spec:
            raise ValueError("No megacomplex defined in model")

        default_megacomplex = spec.pop("default_megacomplex", None)

        if default_megacomplex is None and any(
            "type" not in m for m in spec["megacomplex"].values()
        ):
            raise ValueError(
                "Default megacomplex is not defined in model and "
                "at least one megacomplex does not have a type."
            )

        spec["megacomplex"] = {
            label: m | {"type": default_megacomplex} if "type" not in m else m
            for label, m in spec["megacomplex"].items()
        }

        megacomplex_types = {get_megacomplex(m["type"]) for m in spec["megacomplex"].values()}
        return Model.create_class_from_megacomplexes(megacomplex_types)(**spec)

    def save_model(self, model: Model, file_name: str):
        """Save a :class:`Model` instance to a specification file.

        Parameters
        ----------
        model: Model
            Model instance to save to specs file.
        file_name : str
            File to write the model specs to.
        """
        model_dict = model.as_dict()
        # We replace tuples with strings
        for items in model_dict.values():
            if not isinstance(items, (list, dict)):
                continue
            item_iterator = items if isinstance(items, list) else items.values()
            for item in item_iterator:
                for prop_name, prop in item.items():
                    if isinstance(prop, dict) and any(isinstance(k, tuple) for k in prop):
                        keys = [f"({k[0]}, {k[1]})" for k in prop]
                        item[prop_name] = {f"{k}": v for k, v in zip(keys, prop.values())}
        write_dict(model_dict, file_name=file_name)

    def load_parameters(self, file_name: str) -> Parameters:
        """Load :class:`Parameters` instance from the specification defined in ``file_name``.

        Parameters
        ----------
        file_name: str
            File containing the parameter specification.

        Returns
        -------
        Parameters
        """  # noqa:  D414
        spec = self._load_yml(file_name)

        if isinstance(spec, list):
            return Parameters.from_list(spec)
        else:
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
        spec = self._load_yml(file_name)
        return fromdict(Scheme, spec, folder=Path(file_name).parent)

    def save_scheme(self, scheme: Scheme, file_name: str):
        """Write a :class:`Scheme` instance to a specification file ``file_name``.

        Parameters
        ----------
        scheme: Scheme
            :class:`Scheme` instance to save to file.
        file_name: str
            Path to the file to write the scheme specification to.
        """
        scheme_dict = asdict(scheme, folder=Path(file_name).parent)
        write_dict(scheme_dict, file_name=file_name)

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
        if "number_of_data_points" in spec:
            spec["number_of_residuals"] = spec.pop("number_of_data_points")
        if "number_of_parameters" in spec:
            spec["number_of_free_parameters"] = spec.pop("number_of_parameters")
        return fromdict(Result, spec, folder=result_file_path.parent)

    def save_result(
        self,
        result: Result,
        result_path: str,
        saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT,
    ) -> list[str]:
        """Write a :class:`Result` instance to a specification file and data files.

        Returns a list with paths of all saved items.
        The following files are saved if not configured otherwise:
        * ``result.md``: The result with the model formatted as markdown text.
        * ``result.yml``: Yaml spec file of the result
        * ``model.yml``: Model spec file.
        * ``scheme.yml``: Scheme spec file.
        * ``initial_parameters.csv``: Initially used parameters.
        * ``optimized_parameters.csv``: The optimized parameter as csv file.
        * ``parameter_history.csv``: Parameter changes over the optimization
        * ``optimization_history.csv``: Parsed table printed by the SciPy optimizer
        * ``{dataset_label}.nc``: The result data for each dataset as NetCDF file.

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
        result_folder = result_file_path.parent
        paths = save_result(
            result,
            result_folder,
            format_name="folder",
            saving_options=saving_options,
            allow_overwrite=True,
            used_inside_of_plugin=True,
        )

        model_path = result_folder / "model.yml"
        save_model(result.scheme.model, model_path, allow_overwrite=True)
        paths.append(model_path.as_posix())

        # The source_path attribute of the datasets only gets changed for `result.data`
        # Which why we overwrite the data attribute on a copy of `result.scheme`
        scheme = replace(result.scheme, data=result.data)
        scheme_path = result_folder / "scheme.yml"
        save_scheme(scheme, scheme_path, allow_overwrite=True)
        paths.append(scheme_path.as_posix())

        result_dict = asdict(result, folder=result_folder)
        write_dict(result_dict, file_name=result_file_path)
        paths.append(result_file_path.as_posix())

        return paths

    def _load_yml(self, file_name: str) -> dict[str, Any]:
        return load_dict(file_name, self.format != "yml_str")
