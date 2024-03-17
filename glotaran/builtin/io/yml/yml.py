"""Module containing the YAML Data and Project IO plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.utils.sanitize import sanitize_yaml

if TYPE_CHECKING:
    from typing import Any


@register_project_io(["yml", "yaml", "yml_str"])
class YmlProjectIo(ProjectIoInterface):
    """Plugin for YAML project io."""

    def load_parameters(self, file_name: str) -> Parameters:
        """Load :class:`Parameters` instance from the specification defined in ``file_name``.

        Parameters
        ----------
        file_name: str
            File containing the parameter specification.

        Returns
        -------
        Parameters
        """  # noqa:  D414
        spec = self._load_yml(file_name)

        if isinstance(spec, list):
            return Parameters.from_list(spec)
        return Parameters.from_dict(spec)

    def load_scheme(self, file_name: str) -> Scheme:
        """Load :class:`Scheme` instance from the specification defined in ``file_name``.

        Parameters
        ----------
        file_name: str
            File containing the scheme specification.

        Returns
        -------
        Scheme
        """
        spec = sanitize_yaml(self._load_yml(file_name), do_values=True)
        return Scheme.from_dict(spec)

    #  def save_scheme(self, scheme: Scheme, file_name: str):
    #      """Write a :class:`Scheme` instance to a specification file ``file_name``.
    #
    #      Parameters
    #      ----------
    #      scheme: Scheme
    #          :class:`Scheme` instance to save to file.
    #      file_name: str
    #          Path to the file to write the scheme specification to.
    #      """
    #      scheme_dict = asdict(scheme, folder=Path(file_name).parent)
    #      write_dict(scheme_dict, file_name=file_name)

    def _load_yml(self, file_name: str) -> dict[str, Any]:
        return load_dict(file_name, self.format != "yml_str")
