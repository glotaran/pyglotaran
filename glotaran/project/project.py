"""Opt-in project that records fits and exports results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from glotaran.project.record import FitRecord

if TYPE_CHECKING:
    from glotaran.parameter import Parameters
    from glotaran.project.result import Result
    from glotaran.project.scheme import Scheme
    from glotaran.typing.types import DatasetMappable
    from glotaran.typing.types import StrOrPath

PROJECT_FILE_NAME = "project.gta"
PROJECT_FILE_TEMPLATE = """\
# Description of this project and its data for research data management.
# Every field is optional. pyglotaran reads none of them and copies this file into every export.
title:
description:
data_description:
creators:
  - name:
    orcid:
    affiliation:
keywords: []
license:
related_identifiers:
  - doi:
funding:
"""


class Project:
    """A folder in which fits are recorded and from which results are exported.

    Create an instance with :meth:`Project.start`.
    """

    def __init__(self, folder: Path, results_folder: Path) -> None:
        """Initialize a project with absolute paths; use :meth:`Project.start` instead.

        Parameters
        ----------
        folder : Path
            Absolute path of the project folder, which contains ``project.gta``.
        results_folder : Path
            Absolute path of the folder the records of this instance are written to.
        """
        self.folder = folder
        self.results_folder = results_folder
        self.exports_folder = folder / "exports"
        self.project_file = folder / PROJECT_FILE_NAME

    @classmethod
    def start(cls, folder: StrOrPath = ".", results: StrOrPath = "results") -> Project:
        """Open the project in ``folder``, creating ``project.gta`` if it is missing.

        An existing ``project.gta`` is used as is and never modified, so calling ``start``
        again on the same folder opens the same project. Parent folders are not searched.

        Parameters
        ----------
        folder : StrOrPath
            Project folder, created if it does not exist. Defaults to the working directory.
        results : StrOrPath
            Folder for the records of fits run through this instance, relative to ``folder``.
            Defaults to ``"results"``. Two instances with different ``results`` keep separate
            record collections in one project.

        Returns
        -------
        Project
            The project, with all folders resolved to absolute paths, so that a later change of
            the working directory has no effect.
        """
        project_folder = Path(folder).resolve()
        project_folder.mkdir(parents=True, exist_ok=True)
        try:
            with (project_folder / PROJECT_FILE_NAME).open("x", encoding="utf8") as project_file:
                project_file.write(PROJECT_FILE_TEMPLATE)
        except FileExistsError:
            pass
        return cls(project_folder, (project_folder / results).resolve())

    def optimize(
        self,
        scheme: Scheme,
        parameters: Parameters,
        datasets: DatasetMappable,
        *,
        name: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Result:
        """Run ``scheme.optimize`` with the same arguments and record the fit.

        The record is a folder in :attr:`results_folder` named by the start time of the fit. It
        holds the scheme, the initial and optimized parameters, the cost of every function
        evaluation, a summary of the input data and of the fit, and with ``verbose=True`` the
        parameter values of every function evaluation; it holds no data and no result arrays.
        Failed and interrupted fits are recorded too. ``dry_run=True`` records nothing.

        Parameters
        ----------
        scheme : Scheme
            The scheme to optimize.
        parameters : Parameters
            The initial parameters.
        datasets : DatasetMappable
            The input data, as for :meth:`Scheme.optimize`.
        name : str | None
            Optional label of the record.
        **kwargs : Any
            Keyword arguments of :meth:`Scheme.optimize`.

        Returns
        -------
        Result
            The result, with :attr:`Result.record` referring to the record.
        """
        if kwargs.pop("dry_run", False):
            return scheme.optimize(parameters, datasets, dry_run=True, **kwargs)
        verbose = kwargs.get("verbose", True)
        fit_scheme, optimization = scheme._prepare_optimization(  # noqa: SLF001
            parameters, datasets, **kwargs
        )
        record = FitRecord.start(
            self.results_folder,
            fit_scheme,
            parameters,
            optimization,
            name=name,
            write_parameter_history=verbose,
        )
        if verbose and record is not None:
            print(f"Recording the fit in {record.folder}")  # noqa: T201
        try:
            result = fit_scheme._run_optimization(optimization, parameters)  # noqa: SLF001
        except (Exception, KeyboardInterrupt) as error:
            if record is not None:
                record.finish(optimization, error=error)
            raise
        if record is not None:
            record.finish(optimization, result=result)
            result.record = record.reference
        return result

    def __repr__(self) -> str:
        """Return the project folder and the results folder."""
        return (
            f"Project.start({self.folder.as_posix()!r}, "
            f"results={self.results_folder.as_posix()!r})"
        )
