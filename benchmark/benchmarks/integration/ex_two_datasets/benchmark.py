import pickle
from pathlib import Path

from scipy.optimize import OptimizeResult

from glotaran.analysis.optimize import _create_result
from glotaran.analysis.optimize import optimize
from glotaran.analysis.problem_grouped import GroupedProblem
from glotaran.io import load_dataset
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.project.scheme import Scheme

SCRIPT_DIR = Path(__file__).parent


class IntegrationTwoDatasets:
    """
    Integration test for a two dataset analysis.

    Ref:
    https://github.com/glotaran/pyglotaran-examples/tree/main/pyglotaran_examples/ex_two_datasets
    """

    timeout = 300

    def setup(self):
        dataset1 = load_dataset(SCRIPT_DIR / "data/data1.ascii")
        dataset2 = load_dataset(SCRIPT_DIR / "data/data2.ascii")
        model = load_model(str(SCRIPT_DIR / "models/model.yml"))
        parameters = load_parameters(str(SCRIPT_DIR / "models/parameters.yml"))
        self.scheme = Scheme(
            model,
            parameters,
            {"dataset1": dataset1, "dataset2": dataset2},
            maximum_number_function_evaluations=11,
            non_negative_least_squares=True,
            optimization_method="TrustRegionReflection",
        )
        # Values extracted from a previous run of IntegrationTwoDatasets.time_optimize()
        self.problem = GroupedProblem(self.scheme)
        # pickled OptimizeResult
        with open(SCRIPT_DIR / "data/ls_result.pcl", "rb") as ls_result_file:
            self.ls_result: OptimizeResult = pickle.load(ls_result_file)
        self.free_parameter_labels = [
            "inputs.2",
            "inputs.3",
            "inputs.7",
            "inputs.8",
            "scale.2",
            "rates.k1",
            "rates.k2",
            "rates.k3",
            "irf.center",
            "irf.width",
        ]
        self.termination_reason = "The maximum number of function evaluations is exceeded."

    def time_optimize(self):
        optimize(self.scheme)

    def peakmem_optimize(self):
        optimize(self.scheme)

    def time_create_result(self):
        _create_result(
            self.problem, self.ls_result, self.free_parameter_labels, self.termination_reason
        )

    def peakmem_create_result(self):
        _create_result(
            self.problem, self.ls_result, self.free_parameter_labels, self.termination_reason
        )


if __name__ == "__main__":
    test = IntegrationTwoDatasets()
    test.setup()
    test.time_optimize()
