from pathlib import Path

from glotaran import read_model_from_yaml_file
from glotaran import read_parameters_from_yaml_file
from glotaran.analysis.optimize import optimize
from glotaran.analysis.scheme import Scheme
from glotaran.io import read_data_file

SCRIPT_DIR = Path(__file__).parent


class IntegrationTwoDatasets:
    version = "all-versions"
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        dataset1 = read_data_file(SCRIPT_DIR / "data/data1.ascii")
        dataset2 = read_data_file(SCRIPT_DIR / "data/data2.ascii")
        model = read_model_from_yaml_file(str(SCRIPT_DIR / "models/model.yml"))
        parameters = read_parameters_from_yaml_file(str(SCRIPT_DIR / "models/parameters.yml"))
        self.scheme = Scheme(
            model,
            parameters,
            {"dataset1": dataset1, "dataset2": dataset2},
            maximum_number_function_evaluations=11,
            non_negative_least_squares=True,
            optimization_method="TrustRegionReflection",
        )

    def time_optimize(self):
        optimize(self.scheme)

    def peakmem_optimize(self):
        optimize(self.scheme)
