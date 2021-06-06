from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import TYPE_CHECKING

import yaml

from glotaran.builtin.io.yml.sanatize import sanitize_yaml
from glotaran.io import ProjectIoInterface
from glotaran.io import load_dataset
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import register_project_io
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.model import get_model
from glotaran.parameter import ParameterGroup
from glotaran.project import SavingOptions
from glotaran.project import Scheme

if TYPE_CHECKING:
    from glotaran.model import Model
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

        if self.format == "yml_str":
            spec = yaml.safe_load(file_name)

        else:
            with open(file_name) as f:
                spec = yaml.safe_load(f)

        spec = sanitize_yaml(spec)

        if "type" not in spec:
            raise Exception("Model type not defined")

        model_type = spec["type"]
        del spec["type"]

        model = get_model(model_type)
        return model.from_dict(spec)

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
        nnls = scheme.get("non-negative-least-squares", False)
        nfev = scheme.get("maximum-number-function-evaluations", None)
        ftol = scheme.get("ftol", 1e-8)
        gtol = scheme.get("gtol", 1e-8)
        xtol = scheme.get("xtol", 1e-8)
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
            group_tolerance=group_tolerance,
            optimization_method=optimization_method,
            saving=saving,
        )

    def save_scheme(self, scheme: Scheme, file_name: str):
        _write_dict(file_name, dataclasses.asdict(scheme))

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


def _write_dict(file_name: str, d: dict):
    with open(file_name, "w") as f:
        f.write(yaml.dump(d))
