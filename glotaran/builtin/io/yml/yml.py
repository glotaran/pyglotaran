from __future__ import annotations

import dataclasses
import pathlib

import yaml

from glotaran.deprecation.modules.builtin_io_yml import model_spec_deprecations
from glotaran.io import ProjectIoInterface
from glotaran.io import load_dataset
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.io import register_project_io
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Result
from glotaran.project import SavingOptions
from glotaran.project import Scheme
from glotaran.utils.sanitize import sanitize_yaml


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

    def load_result(self, file_name: str) -> Result:

        spec = self._load_yml(file_name)

        spec["scheme"] = load_scheme(spec["scheme"])
        spec["data"] = spec["scheme"].data

        return Result(**spec)

    def load_parameters(self, file_name: str) -> ParameterGroup:

        spec = self._load_yml(file_name)

        if isinstance(spec, list):
            return ParameterGroup.from_list(spec)
        else:
            return ParameterGroup.from_dict(spec)

    def load_scheme(self, file_name: str) -> Scheme:
        if self.format == "yml_str":
            yml = file_name
        else:
            try:
                with open(file_name) as f:
                    yml = f.read()
            except Exception as e:
                raise OSError(f"Error opening scheme: {e}")

        try:
            scheme = yaml.safe_load(yml)
        except Exception as e:
            raise ValueError(f"Error parsing scheme: {e}")

        if "model" not in scheme:
            raise ValueError("Model file not specified.")

        try:
            model = load_model(scheme["model"])
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

        if "parameters" not in scheme:
            raise ValueError("Parameters file not specified.")

        try:
            parameters = load_parameters(scheme["parameters"])
        except Exception as e:
            raise ValueError(f"Error loading parameters: {e}")

        if "data" not in scheme:
            raise ValueError("No data specified.")

        data = {}
        for label, path in scheme["data"].items():
            data_format = scheme.get("data_format", None)
            path = str(pathlib.Path(path).resolve())

            try:
                data[label] = load_dataset(path, format_name=data_format)
            except Exception as e:
                raise ValueError(f"Error loading dataset '{label}': {e}")

        optimization_method = scheme.get("optimization_method", "TrustRegionReflection")
        nnls = scheme.get("non_negative_least_squares", False)
        nfev = scheme.get("maximum_number_function_evaluations", None)
        ftol = scheme.get("ftol", 1e-8)
        gtol = scheme.get("gtol", 1e-8)
        xtol = scheme.get("xtol", 1e-8)
        group = scheme.get("group", False)
        group_tolerance = scheme.get("group_tolerance", 0.0)
        saving = SavingOptions(**scheme.get("saving", {}))
        return Scheme(
            model=model,
            parameters=parameters,
            data=data,
            non_negative_least_squares=nnls,
            maximum_number_function_evaluations=nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            group=group,
            group_tolerance=group_tolerance,
            optimization_method=optimization_method,
            saving=saving,
        )

    def save_scheme(self, scheme: Scheme, file_name: str):
        file_name = pathlib.Path(file_name)
        scheme_dict = dataclasses.asdict(
            dataclasses.replace(
                scheme,
                model=str(file_name.with_name("model.yml")),
                parameters=str(file_name.with_name("initial_parameters.csv")),
                data={label: str(file_name.with_name(f"{label}.nc")) for label in scheme.data},
            )
        )
        _write_dict(file_name, scheme_dict)

    def save_model(self, model: Model, file_name: str):
        model_dict = model.as_dict()
        # We replace tuples with strings
        for name, items in model_dict.items():
            if not isinstance(items, (list, dict)):
                continue
            item_iterator = items if isinstance(items, list) else items.values()
            for item in item_iterator:
                for prop_name, prop in item.items():
                    if isinstance(prop, dict) and any(isinstance(k, tuple) for k in prop):
                        item[prop_name] = {str(k): v for k, v in prop}
        _write_dict(file_name, model_dict)

    def save_result(self, result: Result, file_name: str):
        options = result.scheme.saving

        result_file_path = pathlib.Path(file_name)
        if result_file_path.exists():
            raise FileExistsError(f"The path '{file_name}' is already existing.")

        scheme_path = result_file_path.with_name("scheme.yml")

        parameters_format = options.parameter_format
        initial_parameters_path = result_file_path.with_name(
            f"initial_parameters.{parameters_format}"
        )
        optimized_parameters_path = result_file_path.with_name(
            f"optimized_parameters.{parameters_format}"
        )

        dataset_format = options.data_format
        data_paths = {
            label: str(result_file_path.with_name(f"{label}.{dataset_format}"))
            for label in result.data
        }

        result_dict = dataclasses.asdict(
            dataclasses.replace(
                result,
                scheme=str(scheme_path),
                initial_parameters=str(initial_parameters_path),
                optimized_parameters=str(optimized_parameters_path),
                data=data_paths,
            )
        )
        del result_dict["additional_penalty"]
        del result_dict["cost"]
        del result_dict["jacobian"]
        del result_dict["covariance_matrix"]
        _write_dict(result_file_path, result_dict)

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
