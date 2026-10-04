"""Opt-in project that records fits and exports results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from glotaran.project.compare import FitComparison
from glotaran.project.compare import compare_fits
from glotaran.project.compare import fit_from_record
from glotaran.project.compare import fit_from_result
from glotaran.project.compare import list_records
from glotaran.project.recompute import recompute
from glotaran.project.record import RECORD_FILE_NAME
from glotaran.project.record import FitRecord
from glotaran.project.record import RecordReference
from glotaran.project.result import Result

if TYPE_CHECKING:
    import pandas as pd

    from glotaran.parameter import Parameters
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

    def recompute(
        self,
        record: str | Path | RecordReference,
        datasets: DatasetMappable,
        *,
        allow_data_mismatch: bool = False,
    ) -> Result:
        """Recompute the results of a recorded fit from its input data.

        Evaluates the recorded scheme once at the recorded optimized parameters with the
        recorded optimizer settings. The data summary of each supplied dataset is compared with
        the recorded one first, and the recomputed cost and RMSE with the recorded values after.
        Records hold no data: regenerate the input data, for example by re-running the
        preprocessing of the notebook. Nothing is recorded.

        Parameters
        ----------
        record : str | Path | RecordReference
            Record id in :attr:`results_folder`, path of a record folder, or
            :attr:`Result.record`.
        datasets : DatasetMappable
            The input data of the fit.
        allow_data_mismatch : bool
            Recompute although the data differ from the recorded data; the differences are
            listed in ``Result.recomputation``. Defaults to ``False``.

        Returns
        -------
        Result
            The recomputed result; ``Result.recomputation`` holds the original fit and the
            reconstruction information. The Jacobian and covariance matrix of the original fit
            are not recomputed.

        Raises
        ------
        GlotaranUserError
            If the data differ from the recorded data and ``allow_data_mismatch`` is ``False``.
        """
        return recompute(
            self._fit_folder(record), datasets, allow_data_mismatch=allow_data_mismatch
        )

    def list_results(
        self,
        *,
        source: str | None = None,
        scheme_source: str | None = None,
        name: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """List the records in :attr:`results_folder`, sorted by the start time of the fits.

        A change in the shape or RMS columns of a dataset shows where its data was replaced.
        Folders without ``record.yml``, such as v0.7 results, are not listed. A record left at
        status ``running`` is incomplete: its fit is still running or never returned to Python.

        Parameters
        ----------
        source : str | None
            Only records whose notebook or script path contains this text.
        scheme_source : str | None
            Only records whose scheme file path contains this text.
        name : str | None
            Only records whose name contains this text.
        status : str | None
            Only records with this status: ``running``, ``success``, ``failed`` or
            ``interrupted``.

        Returns
        -------
        pd.DataFrame
            One row per record, indexed by the record id.
        """
        return list_records(
            self.results_folder,
            source=source,
            scheme_source=scheme_source,
            name=name,
            status=status,
        )

    def compare_results(
        self, a: str | Path | RecordReference | Result, b: str | Path | RecordReference | Result
    ) -> FitComparison:
        """Compare two fits, each given as a record id, a record path or a result.

        Parameters
        ----------
        a : str | Path | RecordReference | Result
            First fit.
        b : str | Path | RecordReference | Result
            Second fit.

        Returns
        -------
        FitComparison
            Parameter fields that differ, the summary statistics of both fits including the
            RMSE per dataset, the diff of the schemes and the differences of the data summaries.
            Changes to weights, penalties, scales and constraints show in the scheme diff or the
            data differences.
        """
        fits = [
            fit_from_result(fit, label)
            if isinstance(fit, Result)
            else fit_from_record(self._fit_folder(fit))
            for fit, label in ((a, "a"), (b, "b"))
        ]
        return compare_fits(*fits)

    def _fit_folder(self, record: str | Path | RecordReference) -> Path:
        """Return the folder of a record given by id, path or reference.

        Raises
        ------
        FileNotFoundError
            If the folder contains no record.
        """
        if isinstance(record, RecordReference):
            folder = record.path
        elif isinstance(record, str) and (self.results_folder / record).is_dir():
            folder = self.results_folder / record
        else:
            folder = Path(record).resolve()
        if not (folder / RECORD_FILE_NAME).is_file():
            msg = f"No record found for {record!r}."
            raise FileNotFoundError(msg)
        return folder

    def __repr__(self) -> str:
        """Return the project folder and the results folder."""
        return (
            f"Project.start({self.folder.as_posix()!r}, "
            f"results={self.results_folder.as_posix()!r})"
        )
