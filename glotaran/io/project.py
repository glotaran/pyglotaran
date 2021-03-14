import dataclasses
import os

from glotaran.plugin_system.data_io_registration import write_dataset
from glotaran.plugin_system.project_io_registration import write_parameters
from glotaran.plugin_system.project_io_registration import write_result
from glotaran.plugin_system.project_io_registration import write_scheme
from glotaran.project import Result


def save_result(result_path: str, result: Result):

    options = result.scheme.saving

    if os.path.exists(result_path):
        raise FileExistsError(f"The path '{result_path}' is already existing.")

    os.makedirs(result_path)

    if options.report:
        md_path = os.path.join(result_path, "result.md")
        with open(md_path, "w") as f:
            f.write(result.markdown())

    scheme_path = os.path.join(result_path, "scheme.yml")
    result_scheme = dataclasses.replace(result.scheme)
    result = dataclasses.replace(result)
    result.scheme = scheme_path

    parameters_format = options.parameter_format

    initial_parameters_path = os.path.join(result_path, f"initial_parameters.{parameters_format}")
    write_parameters(initial_parameters_path, parameters_format, result.initial_parameters)
    result.initial_parameters = initial_parameters_path
    result_scheme.parameters = initial_parameters_path

    optimized_parameters_path = os.path.join(
        result_path, f"optimized_parameters.{parameters_format}"
    )
    write_parameters(optimized_parameters_path, parameters_format, result.optimized_parameters)
    result.optimized_parameters = optimized_parameters_path

    dataset_format = options.data_format
    for label, dataset in result.data.items():
        dataset_path = os.path.join(result_path, f"{label}.{dataset_format}")
        write_dataset(dataset_path, dataset_format, dataset, options)
        result.data[label] = dataset_path
        result_scheme.data[label] = dataset_path

    result_file_path = os.path.join(result_path, "result.yml")
    write_result(result_file_path, "yml", result, options)
    result_scheme.result_path = result_file_path

    write_scheme(scheme_path, "yml", result_scheme)
