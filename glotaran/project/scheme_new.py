from itertools import chain

from pydantic import BaseModel

from glotaran.model import ExperimentModel
from glotaran.model import Library
from glotaran.plugin_system.megacomplex_registration import get_megacomplex


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
