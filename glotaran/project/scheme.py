from __future__ import annotations

from collections import ChainMap
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from glotaran.io import load_dataset
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.optimization.info import calculate_parameter_errors
from glotaran.project.library import ModelLibrary
from glotaran.utils.io import load_datasets

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    import xarray as xr

    from glotaran.parameter import Parameters
    from glotaran.project.result import Result
    from glotaran.typing.types import DatasetMappable


class Scheme(BaseModel):
    """The model of an analysis: a library of elements and the experiments that use them.

    A scheme holds no parameter values, data or optimizer settings; they are arguments of
    :meth:`optimize`. Load a scheme from a file with :func:`glotaran.io.load_scheme` or create
    it from a dictionary with :meth:`from_dict`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    experiments: dict[str, ExperimentModel] = Field(
        description=(
            "The experiments by label. An experiment assigns elements of the library to datasets "
            "that are fitted together."
        )
    )
    library: ModelLibrary = Field(
        description="The elements of the model by label, for use in the experiments."
    )
    source_path: Path | None = Field(
        default=None,
        description="Path to the source file from which this scheme was loaded.",
        exclude=True,
    )

    @classmethod
    def from_dict(cls, spec: dict, source_path: Path | None = None) -> Self:
        """Create a scheme from a dictionary with the structure of a scheme file.

        Parameters
        ----------
        spec : dict
            The scheme with the keys ``library`` and ``experiments``, for example parsed YAML
            or JSON. The key of a rate of a kinetic element is a tuple ``(to, from)`` or the
            text ``"(to, from)"``.
        source_path : Path | None
            Path of the file the dictionary was read from. Defaults to None.

        Returns
        -------
        Self
            The scheme.

        Raises
        ------
        KeyError
            If ``library`` or ``experiments`` is missing, or if a dataset names an element
            that is not in the library.
        pydantic.ValidationError
            If an element, an experiment or a dataset has an unknown field or a field of the
            wrong type.
        """
        library = ModelLibrary.from_dict(spec["library"])
        experiments = {
            k: ExperimentModel.from_dict(library, e) for k, e in spec["experiments"].items()
        }
        for e in experiments.values():
            for d in e.datasets.values():
                if isinstance(d.data, str):
                    d.data = load_dataset(d.data)
        return cls(experiments=experiments, library=library, source_path=source_path)

    def _load_data(self, datasets: Mapping[str, xr.Dataset]) -> None:
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
        ftol: float | None = 1e-8,
        gtol: float | None = 1e-8,
        xtol: float | None = 1e-8,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
        dry_run: bool = False,
        verbose: bool = True,
        raise_exception: bool = False,
    ) -> Result:
        """Fit the scheme to data and return the result.

        The scheme, the parameters and the datasets are copied, so the call does not change
        them.

        Parameters
        ----------
        parameters : Parameters
            The starting values.
        datasets : DatasetMappable
            The data of each dataset of the scheme, as a mapping of dataset label to data.
            A value is an ``xarray.Dataset`` with the variable ``data``, an
            ``xarray.DataArray`` or the path of a data file.
        maximum_number_function_evaluations : int | None
            The limit of function evaluations, or None for the default of SciPy.
            Defaults to None.
        ftol : float | None
            Tolerance for termination by the change of the cost, or None to disable this
            condition. Defaults to 1e-8.
        gtol : float | None
            Tolerance for termination by the norm of the gradient, or None to disable this
            condition. Defaults to 1e-8.
        xtol : float | None
            Tolerance for termination by the change of the parameters, or None to disable
            this condition. Defaults to 1e-8.
        optimization_method : Literal["TrustRegionReflection", "Dogbox", "Levenberg-Marquardt"]
            The method of :func:`scipy.optimize.least_squares`: ``trf``, ``dogbox`` or ``lm``.
            Defaults to "TrustRegionReflection".
        dry_run : bool
            Evaluate the model once at the starting values instead of fitting. The result has
            all arrays and ``optimization_info.success`` is False. Defaults to False.
        verbose : bool
            Print the iteration table of the optimizer, and keep the parameter values of every
            function evaluation in the parameter history. Defaults to True.
        raise_exception : bool
            Raise an error of the optimizer. With False the error becomes a warning and the
            result has ``optimization_info.success`` set to False. Defaults to False.

        Returns
        -------
        Result
            The optimized parameters, the fit statistics and the result arrays of each dataset.

        Raises
        ------
        GlotaranUserError
            If a dataset of the scheme has no data in ``datasets``.
        GlotaranModelIssues
            If the scheme has problems that are found when the fit starts.
        ParameterNotFoundError
            If the scheme refers to a parameter that is not in ``parameters``.

        Notes
        -----
        ``optimization_info.success`` is True when the optimizer returned a solution, also when
        the fit stopped at ``maximum_number_function_evaluations``. Read
        ``optimization_info.termination_reason`` to tell these outcomes apart.

        While the optimizer runs, ``sys.stdout`` of the process is replaced by an object that
        forwards to the original stream, also with ``verbose=False``.

        Examples
        --------
        >>> from glotaran.io import load_dataset, load_parameters, load_scheme
        >>> scheme = load_scheme("scheme.yml")
        >>> parameters = load_parameters("parameters.yml")
        >>> result = scheme.optimize(parameters, {"my_data": load_dataset("data.nc")})
        >>> result.optimization_info.termination_reason
        '`ftol` termination condition is satisfied.'
        """
        scheme, optimization = self._prepare_optimization(
            parameters,
            datasets,
            verbose=verbose,
            raise_exception=raise_exception,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            optimization_method=optimization_method,
        )
        return scheme._run_optimization(optimization, parameters, dry_run=dry_run)  # noqa: SLF001

    def _prepare_optimization(
        self,
        parameters: Parameters,
        datasets: DatasetMappable,
        **optimization_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Scheme, Optimization]:
        """Create the optimization of a copy of this scheme with copies of the datasets.

        The result holds copies, so that a later change of this scheme, the parameters or the
        caller's data does not change it.

        Parameters
        ----------
        parameters : Parameters
            The initial parameters.
        datasets : DatasetMappable
            The datasets.
        **optimization_kwargs : Any
            Keyword arguments of :class:`Optimization`.

        Returns
        -------
        tuple[Scheme, Optimization]
            The copy of the scheme with the data loaded, and its optimization.
        """
        scheme = self.model_copy(deep=True)
        scheme._load_data(  # noqa: SLF001
            {label: data.copy(deep=True) for label, data in load_datasets(datasets).items()}
        )
        optimization = Optimization(
            models=list(scheme.experiments.values()),
            parameters=parameters,
            library=scheme.library,
            **optimization_kwargs,
        )
        return scheme, optimization

    def _run_optimization(
        self, optimization: Optimization, initial_parameters: Parameters, *, dry_run: bool = False
    ) -> Result:
        """Run an optimization created by :meth:`_prepare_optimization` and create its result.

        Parameters
        ----------
        optimization : Optimization
            The optimization of this scheme.
        initial_parameters : Parameters
            The initial parameters, of which the result stores a copy.
        dry_run : bool
            Evaluate the model once at the initial parameters instead of optimizing.

        Returns
        -------
        Result
        """
        # Prevent circular import error
        from glotaran.project.result import Result  # noqa: PLC0415

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
            initial_parameters=initial_parameters.copy(),
            optimized_parameters=optimized_parameters,
            optimizer_settings=optimization.settings,
        )
