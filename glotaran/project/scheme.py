from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict

from glotaran.io import load_dataset
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.project.library import ModelLibrary
from glotaran.project.result import Result

if TYPE_CHECKING:
    import xarray as xr

    from glotaran.parameter import Parameters


class Scheme(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    experiments: dict[str, ExperimentModel]
    library: ModelLibrary

    @classmethod
    def from_dict(cls, spec: dict):
        library = ModelLibrary.from_dict(spec["library"])
        experiments = {
            k: ExperimentModel.from_dict(library, e) for k, e in spec["experiments"].items()
        }
        for e in experiments.values():
            for d in e.datasets.values():
                if isinstance(d.data, str):
                    d.data = load_dataset(d.data)
        return cls(experiments=experiments, library=library)

    def load_data(self, data: dict[str, xr.Dataset]):
        for experiment in self.experiments.values():
            for label, data_model in experiment.datasets.items():
                try:
                    data_model.data = data[label]
                except KeyError as e:
                    raise GlotaranUserError(f"Not data for data model '{label}' provided.") from e

    def optimize(
        self,
        parameters: Parameters,
        verbose: bool = True,
        raise_exception: bool = False,
        maximum_number_function_evaluations: int | None = None,
        add_svd: bool = True,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
        xtol: float = 1e-8,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
        dry_run: bool = False,
    ) -> Result:
        optimization = Optimization(
            list(self.experiments.values()),
            parameters,
            library=self.library,
            verbose=verbose,
            raise_exception=raise_exception,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
            add_svd=add_svd,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            optimization_method=optimization_method,
        )
        optimized_parameters, optimized_data, optimization_result = (
            optimization.dry_run() if dry_run else optimization.run()
        )
        optimization_result.calculate_parameter_errors(optimized_parameters)
        return Result(
            datasets=optimized_data,
            experiments=self.experiments,
            optimization=optimization_result,
            parameters_intitial=parameters,
            parameters_optimized=optimized_parameters,
        )
