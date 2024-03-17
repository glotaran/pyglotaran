"""Convenience script removing data written by the notebooks.

By running this script before ``pytest --nbval docs/source/notebooks/``
the tests pass w/o using ``allow_overwrite=True`` all over the docs.

If you use ``tox`` to run the tests (``tox`` or ``tox -e docs-notebooks``)
this script will be run before the tests.
"""

from __future__ import annotations

import shutil
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "source/notebooks"


def remove_files(path: Path, glob_pattern: str):
    """Removes files with a given pattern from a folder.

    To not accidentally delete files, we only use glob and not rglob.

    Parameters
    ----------
    path : Path
        Path to folder.
    glob_pattern : str
        Glob pattern of the files
    """
    for file in path.glob(glob_pattern):
        file.unlink()


if __name__ == "__main__":
    remove_files(NOTEBOOK_PATH / "quickstart", "*.nc")
    remove_files(NOTEBOOK_PATH / "quickstart/quickstart_project", "*.gta")
    shutil.rmtree(NOTEBOOK_PATH / "quickstart/quickstart_project/data", ignore_errors=True)
    shutil.rmtree(NOTEBOOK_PATH / "quickstart/quickstart_project/results", ignore_errors=True)
