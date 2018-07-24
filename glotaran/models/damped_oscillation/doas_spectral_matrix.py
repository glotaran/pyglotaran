"""Glotaran DOAS Model Spectral Matrix"""

from glotaran.models.spectral_temporal import SpectralMatrix


class DOASSpectralMatrix(SpectralMatrix):

    @property
    def compartment_order(self):
        """Sets the compartment order to map compartment labels to indices in
        the matrix"""
        compartment_order = super(DOASSpectralMatrix, self).compartment_order
        cmplxs = [self.model.megacomplexes[c] for c in self.dataset.megacomplexes]
        oscs = [self.model.oscillations[o] for cmplx in cmplxs
                 for o in cmplx.oscillations]
        oscs = list(set(oscs))
        for osc in oscs:
            compartment_order.append(osc.sin_compartment)
            compartment_order.append(osc.cos_compartment)
        return compartment_order
