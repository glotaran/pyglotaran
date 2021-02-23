"""A little tool to remove empty cells from notebooks.
Since ``nbstripout`` doesn't have this feature yet, we do it ourselves.
See: https://github.com/kynan/nbstripout/issues/131
"""
import json
from pathlib import Path
from typing import List
from typing import Optional

SCRIPT_ROOT_PATH = Path(__file__).parent
NOTEBOOK_BASE_PATH = SCRIPT_ROOT_PATH / "source" / "notebooks"


def strip_empty_cells_from_notebooks(args: Optional[List[str]] = None) -> int:
    """Strips empty cells from notebooks in NOTEBOOK_BASE_PATH."""

    if args is None:
        notebook_paths = NOTEBOOK_BASE_PATH.rglob("*.ipynb")
    else:
        notebook_paths = [Path(arg) for arg in args]

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text())
        originale_nr_of_cells = len(notebook["cells"])
        notebook["cells"] = [cell for cell in notebook["cells"] if cell.get("source", []) != []]
        if originale_nr_of_cells != len(notebook["cells"]):
            print(f"Fixing: {notebook_path}")
            # to ensure an `lf` newline on windows we need to use `.open` instead of `write_text`
            with notebook_path.open(mode="w", encoding="utf8", newline="\n") as f:
                f.write(json.dumps(notebook, indent=1) + "\n")

    return 0


if __name__ == "__main__":
    import sys

    exit(strip_empty_cells_from_notebooks(sys.argv[1:]))
