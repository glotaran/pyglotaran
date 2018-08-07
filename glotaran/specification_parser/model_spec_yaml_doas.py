from glotaran.models.damped_oscillation import (
    DOASModel,
    DOASMegacomplex,
    Oscillation)

from .model_spec_yaml import (Keys,
                              register_model_parser)
from .model_spec_yaml_kinetic import KineticKeys, KineticModelParser
from .utils import get_keys_from_object


class DOASKeys:
    OSCILLATIONS = "oscillations"
    FREQUENCY = "frequency"
    RATE = "rate"


class DOASModelParser(KineticModelParser):
    def get_model(self):
        return DOASModel

    def get_additionals(self):
        super(DOASModelParser, self).get_additionals()
        self.get_oscillations()

    def get_megacomplexes(self):
        for cmplx in self.spec[Keys.MEGACOMPLEXES]:
            (label, mat, osc) = get_keys_from_object(cmplx,
                                                     [Keys.LABEL,
                                                      KineticKeys.K_MATRICES,
                                                      DOASKeys.OSCILLATIONS,
                                                      ])
            mat = [] if mat is None else mat
            osc = [] if osc is None else osc
            self.model.add_megacomplex(DOASMegacomplex(label, mat, osc))

    def get_oscillations(self):
        if DOASKeys.OSCILLATIONS not in self.spec:
            return
        for osc in self.spec[DOASKeys.OSCILLATIONS]:
            (label, comp, freq, rate) = get_keys_from_object(osc,
                                                             [Keys.LABEL,
                                                              Keys.COMPARTMENT,
                                                              DOASKeys.FREQUENCY,
                                                              DOASKeys.RATE,
                                                              ])
            self.model.add_oscillation(Oscillation(label, comp, freq, rate))


register_model_parser("dampened_oscillation", DOASModelParser)
