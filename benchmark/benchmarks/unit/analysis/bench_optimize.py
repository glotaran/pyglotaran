import numpy as np
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import MultichannelMulticomponentDecay
from glotaran.project import Scheme


class BenchmarkOptimize:
    """
    Integration test for a two dataset analysis.

    Ref:
    https://github.com/glotaran/pyglotaran-examples/tree/main/pyglotaran_examples/ex_two_datasets
    """

    timeout = 300
    params = (
        # index_dependent
        [True, False],
        # grouped
        [True, False],
        # weight
        [True, False],
    )
    param_names = ["index_dependent", "grouped", "weight"]

    def setup(self, index_dependent, grouped, weight):
        suite = MultichannelMulticomponentDecay
        model = suite.model
        # 0.4.0 API compat
        model.is_grouped = grouped

        model.megacomplex["m1"].is_index_dependent = index_dependent

        sim_model = suite.sim_model
        suite.sim_model.megacomplex["m1"].is_index_dependent = index_dependent

        wanted_parameters = suite.wanted_parameters

        initial_parameters = suite.initial_parameters
        model.dataset["dataset1"].fill(model, initial_parameters)

        if hasattr(suite, "global_axis"):
            axes_dict = {
                "global": getattr(suite, "global_axis"),
                "model": getattr(suite, "model_axis"),
            }
        else:
            # 0.4.0 API compat
            axes_dict = {
                "e": getattr(suite, "e_axis"),
                "c": getattr(suite, "c_axis"),
            }

        dataset = simulate(sim_model, "dataset1", wanted_parameters, axes_dict)

        if weight:
            dataset["weight"] = xr.DataArray(
                np.ones_like(dataset.data) * 0.5, coords=dataset.data.coords
            )

        data = {"dataset1": dataset}

        self.scheme = Scheme(
            model=model,
            parameters=initial_parameters,
            data=data,
            maximum_number_function_evaluations=10,
            group_tolerance=0.1,
            optimization_method="TrustRegionReflection",
        )
        # 0.4.0 API compat
        if hasattr(self.scheme, "group"):
            self.scheme.group = grouped

    def time_optimize(self, index_dependent, grouped, weight):
        optimize(self.scheme)
