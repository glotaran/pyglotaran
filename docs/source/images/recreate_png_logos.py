"""Helper script to render svg logos to png.

This script uses https://github.com/shakiba/svgexport

``npm install svgexport -g``
"""

from __future__ import annotations

import subprocess
from pathlib import Path

image_folder = Path(__file__).parent
output_folder = image_folder / "png"

output_folder.mkdir(parents=True, exist_ok=True)


def render_svg(glob_pattern: str, size: int) -> None:
    """Render svg file tp png file with size ``size x size``.

    Parameters
    ----------
    glob_pattern: str
        Pattern to find svg files.
    size: int
        Size of the resulting png image.
    """

    for logo in image_folder.glob(glob_pattern):
        input_file = logo.resolve().as_posix()
        output_file = (output_folder / f"{logo.stem}_{size}x{size}.png").resolve().as_posix()
        subprocess.run(f"svgexport {input_file} {output_file} {size}:", shell=True)


if __name__ == "__main__":
    for size in (16, 24, 32, 48):
        render_svg("pyglotaran_favicon*.svg", size)
    for size in (64, 128, 256, 512):
        render_svg("pyglotaran_logo*.svg", size)
