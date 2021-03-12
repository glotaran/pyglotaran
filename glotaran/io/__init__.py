"""Functions for data IO"""

from glotaran.plugin_system.data_io_registration import get_dataloader
from glotaran.plugin_system.data_io_registration import get_datawriter
from glotaran.plugin_system.data_io_registration import load_dataset
from glotaran.plugin_system.data_io_registration import register_data_io
from glotaran.plugin_system.data_io_registration import write_dataset

from .decorator import register_project_io
from .interface import DataIoInterface
from .interface import ProjectIoInterface
from .prepare_dataset import prepare_time_trace_dataset
from .project import save_result
from .register import load_model
from .register import load_parameters
from .register import load_scheme
from .register import write_parameters
