from pathlib import Path

from glotaran.analysis.optimize import optimize
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

    def time_optimize(self):
        optimize(self.scheme)

    def peakmem_optimize(self):
        optimize(self.scheme)
