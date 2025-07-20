from __future__ import annotations

from collections import ChainMap
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from glotaran.io import load_dataset
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.optimization.result import calculate_parameter_errors
from glotaran.project.library import ModelLibrary
from glotaran.utils.io import DatasetMapping
from glotaran.utils.io import load_datasets

if TYPE_CHECKING:
    from typing_extensions import Self

    from glotaran.parameter import Parameters
    from glotaran.project.result import Result
    from glotaran.typing.types import DatasetMappable


class Scheme(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    experiments: dict[str, ExperimentModel]
    library: ModelLibrary
    source_path: Path | None = Field(
        default=None,
        description="Path to the source file from which this scheme was loaded.",
        exclude=True,
    )

    @classmethod
    def from_dict(cls, spec: dict, source_path: Path | None = None) -> Self:
        library = ModelLibrary.from_dict(spec["library"])
        experiments = {
            k: ExperimentModel.from_dict(library, e) for k, e in spec["experiments"].items()
        }
        for e in experiments.values():
            for d in e.datasets.values():
                if isinstance(d.data, str):
                    d.data = load_dataset(d.data)
        return cls(experiments=experiments, library=library, source_path=source_path)

    def _load_data(self, datasets: DatasetMapping) -> None:
        try:
            for experiment in self.experiments.values():
                for label, data_model in experiment.datasets.items():
                    data_model.data = datasets[label]
        except KeyError as e:
            msg = f"Not data for data model '{label}' provided."
            raise GlotaranUserError(msg) from e

    @property
    def dataset_paths(self) -> dict[str, str]:
        """Paths to all the datasets."""
        return dict(
            ChainMap(*(experiment.dataset_paths for experiment in self.experiments.values()))
        )

    def optimize(
        self,
        parameters: Parameters,
        datasets: DatasetMappable,
        *,
        maximum_number_function_evaluations: int | None = None,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
        xtol: float = 1e-8,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
        add_svd: bool = True,
        dry_run: bool = False,
        verbose: bool = True,
        raise_exception: bool = False,
    ) -> Result:
        # Prevent circular import error
        from glotaran.project.result import Result  # noqa: PLC0415

        self._load_data(load_datasets(datasets))
        optimization = Optimization(
            models=list(self.experiments.values()),
            parameters=parameters,
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
        optimized_parameters, optimized_data, optimization_info = (
            optimization.dry_run() if dry_run else optimization.run()
        )
        calculate_parameter_errors(
            optimization_info=optimization_info, parameters=optimized_parameters
        )
        return Result(
            optimization_results=optimized_data,
            scheme=self,
            optimization_info=optimization_info,
            initial_parameters=parameters,
            optimized_parameters=optimized_parameters,
        )
