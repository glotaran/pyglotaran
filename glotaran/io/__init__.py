"""Functions for data IO

Note:
-----
Since Io functionality is purely plugin based this package mostly
reexports functions from the pluginsystem from a common place.
"""

from glotaran.io.interface import DataIoInterface
from glotaran.io.interface import ProjectIoInterface
from glotaran.io.prepare_dataset import prepare_time_trace_dataset
from glotaran.plugin_system.data_io_registration import data_io_plugin_table
from glotaran.plugin_system.data_io_registration import get_dataloader
from glotaran.plugin_system.data_io_registration import get_datasaver
from glotaran.plugin_system.data_io_registration import load_dataset
from glotaran.plugin_system.data_io_registration import register_data_io
from glotaran.plugin_system.data_io_registration import save_dataset
from glotaran.plugin_system.data_io_registration import show_data_io_method_help
from glotaran.plugin_system.project_io_registration import get_project_io_method
from glotaran.plugin_system.project_io_registration import load_model
from glotaran.plugin_system.project_io_registration import load_parameters
from glotaran.plugin_system.project_io_registration import load_result
from glotaran.plugin_system.project_io_registration import load_scheme
from glotaran.plugin_system.project_io_registration import project_io_plugin_table
from glotaran.plugin_system.project_io_registration import register_project_io
from glotaran.plugin_system.project_io_registration import save_model
from glotaran.plugin_system.project_io_registration import save_parameters
from glotaran.plugin_system.project_io_registration import save_result
from glotaran.plugin_system.project_io_registration import save_scheme
from glotaran.plugin_system.project_io_registration import show_project_io_method_help


def __getattr__(attribute_name: str):
    from glotaran.deprecation.deprecation_utils import module_attribute
    from glotaran.deprecation.deprecation_utils import warn_deprecated

    if attribute_name == "read_data_file":
        warn_deprecated(
            deprecated_qual_name_usage="glotaran.io.read_data_file",
            new_qual_name_usage="glotaran.io.load_dataset",
            to_be_removed_in_version="0.6.0",
            check_qualnames=(False, True),
        )

        return module_attribute(
            module_qual_name="glotaran.io",
            attribute_name="load_dataset",
        )

    raise AttributeError(f"module {__name__} has no attribute {attribute_name}")
