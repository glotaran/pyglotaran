from itertools import chain
from typing import Literal

from pydantic import BaseModel

from glotaran.model import ExperimentModel
from glotaran.model import Library
from glotaran.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.plugin_system.megacomplex_registration import get_megacomplex
from glotaran.project.result_new import Result


class Scheme(BaseModel):
    experiments: list[ExperimentModel]
    library: Library

    @classmethod
    def from_dict(cls, spec: dict):
        megacomplex_types = {
            get_megacomplex(m["type"])
            for e in spec["experiments"]
            for d in e
            for m in chain(d["megacomplex"], d.get("global_megacomples", []))
        }
        library = Library.create_for_megacomplexes(megacomplex_types)(**spec["library"])
        experiments = {
            k: ExperimentModel.from_dict(library, v) for k, v in spec["experiment"].items()
        }
        return cls(experiments=experiments, library=library)

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
    ) -> Result:
        optimized_parameters, optimized_data, optimization_result = Optimization(
            self.experiments,
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
        return Result(
            data=optimized_data,
            optimization=optimization_result,
            parameters_intitial=parameters,
            parameters_optimized=optimized_parameters,
            scheme=self,
        )
