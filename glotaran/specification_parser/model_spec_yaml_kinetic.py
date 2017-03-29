from .model_spec_yaml import (Keys,
                              ModelSpecParser,
                              register_model_parser)
from ast import literal_eval as make_tuple
from collections import OrderedDict
from .utils import get_keys_from_object, retrieve_optional
from glotaran.models.spectral_temporal import (KMatrix,
                                               KineticDatasetDescriptor,
                                               KineticMegacomplex,
                                               KineticModel,
                                               GaussianIrf,
                                               SpectralShapeGaussian)


class KineticKeys(object):
    AMPLITUDE = "amplitude"
    CENTER = 'center'
    CENTER_DISPERSION = 'center_dispersion'
    GAUSSIAN = 'gaussian'
    IRF = 'irf'
    K_MATRICES = "k_matrices"
    LOCATION = "location"
    MATRIX = 'matrix'
    NORMALIZE = 'normalize'
    SCALE = 'scale'
    SHAPE = 'shapes'
    WIDTH = 'width'
    WIDTH_DISPERSION = 'width_dispersion'


class KineticModelParser(ModelSpecParser):
    def get_model(self):
        return KineticModel

    def get_dataset_descriptor(self, label, initial_concentration,
                               megacomplexes, megacomplex_scalings,
                               dataset_scaling, compartment_scalings,
                               dataset_spec):
        irf = dataset_spec[KineticKeys.IRF] if KineticKeys.IRF \
            in dataset_spec else None

        return KineticDatasetDescriptor(label, initial_concentration,
                                        megacomplexes, megacomplex_scalings,
                                        dataset_scaling, compartment_scalings,
                                        irf)

    def get_megacomplexes(self):
        if KineticKeys.K_MATRICES not in self.spec:
            raise Exception("No k-matrices defined")
        for km in self.spec[KineticKeys.K_MATRICES]:
            m = OrderedDict()
            for i in km[KineticKeys.MATRIX]:
                m[make_tuple(i)] = km[KineticKeys.MATRIX][i]
            self.model.add_k_matrix(KMatrix(km[Keys.LABEL], m,
                                            self.model.compartments))
        for cmplx in self.spec[Keys.MEGACOMPLEXES]:
            (label, mat) = get_keys_from_object(cmplx, [Keys.LABEL,
                                                        KineticKeys.K_MATRICES]
                                                )
            self.model.add_megacomplex(KineticMegacomplex(label, mat))

    def get_additionals(self):
        self.get_irfs()
        self.get_shapes()

    def get_irfs(self):
        if KineticKeys.IRF in self.spec:
            for irf in self.spec[KineticKeys.IRF]:
                (label, type) = get_keys_from_object(irf, [Keys.LABEL,
                                                           Keys.TYPE])
                if type == KineticKeys.GAUSSIAN:
                    (center, width) = get_keys_from_object(irf,
                                                           [KineticKeys.CENTER,
                                                            KineticKeys.WIDTH],
                                                           start=2)

                    center_disp = retrieve_optional(irf,
                                                    KineticKeys.
                                                    CENTER_DISPERSION,
                                                    4, [])

                    width_disp = retrieve_optional(irf,
                                                   KineticKeys.
                                                   WIDTH_DISPERSION,
                                                   5, [])

                    scale = retrieve_optional(irf, KineticKeys.SCALE, 6, [])

                    norm = retrieve_optional(irf, KineticKeys.NORMALIZE, 7,
                                             True)

                    self.model.add_irf(GaussianIrf(label, center, width,
                                       center_dispersion=center_disp,
                                       width_dispersion=width_disp,
                                       scale=scale,
                                       normalize=norm))

    def get_shapes(self):
        if KineticKeys.SHAPE in self.spec:
            for shape in self.spec[KineticKeys.SHAPE]:
                (label, type) = get_keys_from_object(shape, [Keys.LABEL,
                                                             Keys.TYPE])
                if type == KineticKeys.GAUSSIAN:
                    (amp, loc, width) = \
                        get_keys_from_object(shape,
                                             [KineticKeys.AMPLITUDE,
                                              KineticKeys.LOCATION,
                                              KineticKeys.WIDTH],
                                             start=2)

                    self.model.add_shape(SpectralShapeGaussian(label,
                                                               amp,
                                                               loc,
                                                               width))

register_model_parser("kinetic", KineticModelParser)
