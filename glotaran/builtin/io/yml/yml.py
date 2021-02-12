import yaml

from glotaran.analysis.result import Result
from glotaran.io import io
from glotaran.model import Model
from glotaran.parameter import Parameter
from glotaran.register import register

from .sanatize import sanitize_yaml


@io(["yml", "yaml", "yml_str"])
class YamlIo:
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

        if not register.known_model(model_type):
            raise Exception(f"Unknown model type '{model_type}'.")

        model = register.get_model(model_type)
        try:
            return model.from_dict(spec)
        except Exception as e:
            raise e

    def write_model(fmt: str, file_name: str, model: Model):
        pass

    def read_parameter(fmt: str, file_name: str) -> Parameter:
        pass

    def write_parameter(fmt: str, file_name: str, parameter: Parameter):
        pass

    def read_result(fmt: str, result_name: str) -> Result:
        pass

    def write_result(fmt: str, result_name: str, result: Result):
        pass
