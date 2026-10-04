"""Opt-in project that records fits and exports results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    def __repr__(self) -> str:
        """Return the project folder and the results folder."""
        return (
            f"Project.start({self.folder.as_posix()!r}, "
            f"results={self.results_folder.as_posix()!r})"
        )
