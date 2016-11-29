from .model_spec_yaml import (ModelKeys,
                              ModelSpecParser,
                              is_compact,
                              register_model_parser)
from ast import literal_eval as make_tuple
from glotaran_models.kinetic import (KMatrix,
                                     KineticDatasetDescriptor,
                                     KineticMegacomplex,
                                     KineticModel,
                                     GaussianIrf)


class KineticModelKeys(object):
    K_MATRICES = "k_matrices"
    MATRIX = 'matrix'
    IRF = 'irf'


class IrfTypes:
    GAUSSIAN = 'gaussian'


class GaussianIrfKeys:
    CENTER = 'center'
    WIDTH = 'width'
    CENTER_DISPERSION = 'center_dispersion'
    WIDTH_DISPERSION = 'width_dispersion'
    SCALE = 'scale'
    NORMALIZE = 'normalize'


class KineticModelParser(ModelSpecParser):
    def get_model(self):
        return KineticModel

    def get_dataset_descriptor(self, label, initial_concentration,
                               megacomplexes, megacomplex_scalings,
                               dataset, dataset_scaling, dataset_spec):
        try:
            irf = dataset_spec[KineticModelKeys.IRF]
        except:
            irf = None

        return KineticDatasetDescriptor(label, initial_concentration,
                                        megacomplexes, megacomplex_scalings,
                                        dataset, dataset_scaling, irf)

    def get_megacomplexes(self):
        if KineticModelKeys.K_MATRICES not in self.spec:
            raise Exception("No k-matrices defined")
        for km in self.spec[KineticModelKeys.K_MATRICES]:
            m = {}
            for i in km[KineticModelKeys.MATRIX]:
                m[make_tuple(i)] = km[KineticModelKeys.MATRIX][i]
            self.model.add_k_matrix(KMatrix(km[ModelKeys.LABEL], m))
        for cmplx in self.spec[ModelKeys.MEGACOMPLEXES]:
            l = ModelKeys.LABEL
            km = KineticModelKeys.K_MATRICES
            if isinstance(cmplx, list):
                l = 0
                km = 1
            self.model.add_megacomplex(KineticMegacomplex(cmplx[l],
                                       cmplx[km]))

    def get_additionals(self):
        self.get_irfs()

    def get_irfs(self):
        if KineticModelKeys.IRF in self.spec:
            for irf in self.spec[KineticModelKeys.IRF]:
                compact = is_compact(irf)

                lb = ModelKeys.LABEL
                tp = ModelKeys.TYPE
                if compact:
                    lb = 0
                    tp = 1
                if irf[tp] == IrfTypes.GAUSSIAN:
                    ct = GaussianIrfKeys.CENTER
                    wt = GaussianIrfKeys.WIDTH
                    cd = GaussianIrfKeys.CENTER_DISPERSION
                    wd = GaussianIrfKeys.WIDTH_DISPERSION
                    if compact:
                        ct = 2
                        wt = 3
                        cd = 4
                        wd = 5

                    scale = retrieve_optional(irf, GaussianIrfKeys.SCALE, 6,
                                              [])
                    norm = retrieve_optional(irf, GaussianIrfKeys.NORMALIZE, 7,
                                             True)

                    self.model.add_irf(GaussianIrf(irf[lb], irf[ct], irf[wt],
                                       center_dispersion=irf[cd],
                                       width_dispersion=irf[wd],
                                       scale=scale,
                                       normalize=norm))


def retrieve_optional(obj, key, index, default):
    compact = is_compact(obj)
    if compact:
        if len(obj) <= index:
            return default
        return obj[index]
    else:
        if key in obj:
            return obj[key]
        return default

register_model_parser("kinetic", KineticModelParser)
