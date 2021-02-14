import pathlib
from dataclasses import asdict

import yaml

from glotaran.io import Io
from glotaran.io import load_dataset
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import register_io
from glotaran.model import Model
from glotaran.model import get_model
from glotaran.parameter import ParameterGroup
from glotaran.project import Result
from glotaran.project import SavingOptions
from glotaran.project import Scheme

from .sanatize import sanitize_yaml


@register_io(["yml", "yaml", "yml_str"])
class YmlIo(Io):
    @staticmethod
    def read_model(fmt: str, file_name: str) -> Model:
        """parse_yaml_file reads the given file and parses its content as YML.

        Parameters
        ----------
        filename : str
            filename is the of the file to parse.

        Returns
        -------
        content : Dict
            The content of the file as dictionary.
        """

        if fmt == "yml_str":
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

    @staticmethod
    def read_parameters(fmt: str, file_name: str) -> ParameterGroup:

        if fmt == "yml_str":
            spec = yaml.safe_load(file_name)
        else:
            with open(file_name) as f:
                spec = yaml.safe_load(f)

        if isinstance(spec, list):
            return ParameterGroup.from_list(spec)
        else:
            return ParameterGroup.from_dict(spec)

    @staticmethod
    def read_scheme(fmt: str, file_name: str) -> Scheme:
        if fmt == "yml_str":
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
            fmt = scheme.get("data_format", None)
            path = pathlib.Path(path)

            try:
                data[label] = load_dataset(path, fmt=fmt)
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

    @staticmethod
    def write_scheme(fmt: str, file_name: str, scheme: Scheme):
        _write_dict(file_name, asdict(scheme))

    @staticmethod
    def write_result(fmt: str, file_name: str, saving_options: SavingOptions, result: Result):
        _write_dict(file_name, asdict(result))


def _write_dict(file_name: str, d: dict):
    with open(file_name, "w") as f:
        f.write(yaml.dump(d))
