from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.builtin.io.yml.utils import write_dict
from glotaran.deprecation.modules.builtin_io_yml import model_spec_deprecations
from glotaran.deprecation.modules.builtin_io_yml import scheme_spec_deprecations
from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.io import ProjectIoInterface
from glotaran.io import SavingOptions
from glotaran.io import register_project_io
from glotaran.io import save_model
from glotaran.io import save_result
from glotaran.io import save_scheme
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project.dataclass_helpers import asdict
from glotaran.project.dataclass_helpers import fromdict
from glotaran.project.project import Result
from glotaran.project.scheme import Scheme
from glotaran.utils.sanitize import sanitize_yaml

if TYPE_CHECKING:
    from typing import Any


@register_project_io(["yml", "yaml", "yml_str"])
class YmlProjectIo(ProjectIoInterface):
    def load_model(self, file_name: str) -> Model:
        """parse_yaml_file reads the given file and parses its content as YML.

        Parameters
        ----------
        filename : str
            filename is the of the file to parse.

        Returns
        -------
        Model
            The content of the file as dictionary.
        """

        spec = self._load_yml(file_name)

        model_spec_deprecations(spec)

        spec = sanitize_yaml(spec)

        default_megacomplex = spec.get("default_megacomplex")

        if default_megacomplex is None and any(
            "type" not in m for m in spec["megacomplex"].values()
        ):
            raise ValueError(
                "Default megacomplex is not defined in model and "
                "at least one megacomplex does not have a type."
            )

        if "megacomplex" not in spec:
            raise ValueError("No megacomplex defined in model")

        return Model.from_dict(spec, megacomplex_types=None, default_megacomplex_type=None)

    def save_model(self, model: Model, file_name: str):
        """Save a Model instance to a spec file.
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

    def load_parameters(self, file_name: str) -> ParameterGroup:
        """Create a ParameterGroup instance from the specs defined in a file.
        Parameters
        ----------
        file_name : str
            File containing the parameter specs.
        Returns
        -------
        ParameterGroup
            ParameterGroup instance created from the file.
        """

        spec = self._load_yml(file_name)

        if isinstance(spec, list):
            return ParameterGroup.from_list(spec)
        else:
            return ParameterGroup.from_dict(spec)

    def load_scheme(self, file_name: str) -> Scheme:
        spec = self._load_yml(file_name)
        scheme_spec_deprecations(spec)
        return fromdict(Scheme, spec, folder=Path(file_name).parent)

    def save_scheme(self, scheme: Scheme, file_name: str):
        scheme_dict = asdict(scheme, folder=Path(file_name).parent)
        write_dict(scheme_dict, file_name=file_name)

    def load_result(self, result_path: str) -> Result:
        """Create a :class:`Result` instance from the specs defined in a file.

        Parameters
        ----------
        result_path : str | PathLike[str]
            Path containing the result data.

        Returns
        -------
        Result
            :class:`Result` instance created from the saved format.
        """
        spec = self._load_yml(result_path)
        return fromdict(Result, spec, folder=Path(result_path).parent)

    def save_result(
        self,
        result: Result,
        result_path: str,
        saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT,
    ) -> list[str]:
        """Write a :class:`Result` instance to a spec file and data files.

        Returns a list with paths of all saved items.
        The following files are saved if not configured otherwise:
        * ``result.md``: The result with the model formatted as markdown text.
        * ``result.yml``: Yaml spec file of the result
        * ``model.yml``: Model spec file.
        * ``scheme.yml``: Scheme spec file.
        * ``initial_parameters.csv``: Initially used parameters.
        * ``optimized_parameters.csv``: The optimized parameter as csv file.
        * ``parameter_history.csv``: Parameter changes over the optimization
        * ``{dataset_label}.nc``: The result data for each dataset as NetCDF file.

        Parameters
        ----------
        result : Result
            :class:`Result` instance to write.
        result_path : str | PathLike[str]
            Path to write the result data to.
        saving_options : SavingOptions
            Options for saving the the result.

        Returns
        -------
        list[str]
            List of file paths which were created.
        """
        result_folder = Path(result_path).parent
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

        scheme_path = result_folder / "scheme.yml"
        save_scheme(result.scheme, scheme_path, allow_overwrite=True)
        paths.append(scheme_path.as_posix())

        result_dict = asdict(result, folder=result_folder)
        write_dict(result_dict, file_name=result_path)
        paths.append(result_path)

        return paths

    def _load_yml(self, file_name: str) -> dict[str, Any]:
        return load_dict(file_name, self.format != "yml_str")
