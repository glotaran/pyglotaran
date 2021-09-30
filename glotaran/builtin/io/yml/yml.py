from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import TYPE_CHECKING

import yaml

from glotaran.deprecation.modules.builtin_io_yml import model_spec_deprecations
from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme
from glotaran.project.dataclasses import asdict
from glotaran.project.dataclasses import fromdict
from glotaran.utils.sanitize import sanitize_yaml

if TYPE_CHECKING:
    from glotaran.project import Result


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

        default_megacomplex = spec.get("default-megacomplex")

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
        for name, items in model_dict.items():
            if not isinstance(items, (list, dict)):
                continue
            item_iterator = items if isinstance(items, list) else items.values()
            for item in item_iterator:
                for prop_name, prop in item.items():
                    if isinstance(prop, dict) and any(isinstance(k, tuple) for k in prop):
                        keys = [f"({k[0]}, {k[1]})" for k in prop]
                        item[prop_name] = {f"{k}": v for k, v in zip(keys, prop.values())}
        _write_dict(file_name, model_dict)

    def load_parameters(self, file_name: str) -> ParameterGroup:

        if self.format == "yml_str":
            spec = yaml.safe_load(file_name)
        else:
            with open(file_name) as f:
                spec = yaml.safe_load(f)

        if isinstance(spec, list):
            return ParameterGroup.from_list(spec)
        else:
            return ParameterGroup.from_dict(spec)

    def load_scheme(self, file_name: str) -> Scheme:
        spec = self._load_yml(file_name)
        file_path = pathlib.Path(file_name)
        return fromdict(Scheme, spec, folder=file_path.parent)

    def save_scheme(self, scheme: Scheme, file_name: str):
        file_name = pathlib.Path(file_name)
        scheme_dict = asdict(scheme)
        _write_dict(file_name, scheme_dict)

    def save_result(self, result: Result, result_path: str):
        options = result.scheme.saving

        if os.path.exists(result_path):
            raise FileExistsError(f"The path '{result_path}' is already existing.")

        os.makedirs(result_path)

        if options.report:
            md_path = os.path.join(result_path, "result.md")
            with open(md_path, "w") as f:
                f.write(str(result.markdown()))

        scheme_path = os.path.join(result_path, "scheme.yml")
        result_scheme = dataclasses.replace(result.scheme)
        result_scheme.model = result_scheme.model.markdown()
        result = dataclasses.replace(result)
        result.scheme = scheme_path

        parameters_format = options.parameter_format

        initial_parameters_path = os.path.join(
            result_path, f"initial_parameters.{parameters_format}"
        )
        save_parameters(result.initial_parameters, initial_parameters_path, parameters_format)
        result.initial_parameters = initial_parameters_path
        result_scheme.parameters = initial_parameters_path

        optimized_parameters_path = os.path.join(
            result_path, f"optimized_parameters.{parameters_format}"
        )
        save_parameters(result.optimized_parameters, optimized_parameters_path, parameters_format)
        result.optimized_parameters = optimized_parameters_path

        dataset_format = options.data_format
        for label, dataset in result.data.items():
            dataset_path = os.path.join(result_path, f"{label}.{dataset_format}")
            save_dataset(dataset, dataset_path, dataset_format, saving_options=options)
            result.data[label] = dataset_path
            result_scheme.data[label] = dataset_path

        result_file_path = os.path.join(result_path, "result.yml")
        _write_dict(result_file_path, dataclasses.asdict(result))
        result_scheme.result_path = result_file_path

        self.save_scheme(scheme=result_scheme, file_name=scheme_path)

    def _load_yml(self, file_name: str) -> dict:
        if self.format == "yml_str":
            spec = yaml.safe_load(file_name)
        else:
            with open(file_name) as f:
                spec = yaml.safe_load(f)
        return spec


def _write_dict(file_name: str, d: dict):
    with open(file_name, "w") as f:
        f.write(yaml.dump(d))
