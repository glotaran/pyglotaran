import yaml

from glotaran.io import register_io
from glotaran.model import Model
from glotaran.model import get_model
from glotaran.model import known_model
from glotaran.parameter import ParameterGroup

from .sanatize import sanitize_yaml


@register_io(["yml", "yaml", "yml_str"])
class YmlIo:
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

        if not known_model(model_type):
            raise Exception(f"Unknown model type '{model_type}'.")

        model = get_model(model_type)
        try:
            return model.from_dict(spec)
        except Exception as e:
            raise e

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
