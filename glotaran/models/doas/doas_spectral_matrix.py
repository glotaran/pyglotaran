"""Glotaran DOAS Model Spectral Matrix"""

import numpy as np

from glotaran.models.spectral_temporal.spectral_matrix import calculate_spectral_matrix

from .doas_matrix import _collect_oscillations


def calculate_doas_spectral_matrix(dataset, axis):

    oscillations = _collect_oscillations(dataset)
    all_oscillations = []
    for osc in oscillations:
        all_oscillations.append(osc)
        all_oscillations.append(osc)
    matrix = np.ones((axis.size, len(all_oscillations)))
    for i, osc in enumerate(all_oscillations):
        if osc.label not in dataset.shapes:
            raise Exception(f'No shape for oscillation "{osc.label}"')
        shapes = dataset.shapes[osc.label]
        if not isinstance(shapes, list):
            shapes = [shapes]
        for shape in shapes:
            matrix[:, i] += shape.calculate(axis)
    return matrix
#
#
#  class DOASSpectralMatrix(SpectralMatrix):
#
#      def collect_shapes(self):
#          super(DOASSpectralMatrix, self).collect_shapes()
#          cmplxs = [self.model.megacomplexes[c] for c in self.dataset.megacomplexes]
#          oscs = [self.model.oscillations[o] for cmplx in cmplxs
#                  for o in cmplx.oscillations]
#          oscs = list(set(oscs))
#          for osc in oscs:
#              if osc in self.shapes:
#                  shape = self.shapes[osc]
#                  self.shapes[f"{osc}_sin"] = shape
#                  self.shapes[f"{osc}_cos"] = shape
#
#      @property
#      def compartment_order(self):
#          """Sets the compartment order to map compartment labels to indices in
#          the matrix"""
#          compartment_order = super(DOASSpectralMatrix, self).compartment_order
#          cmplxs = [self.model.megacomplexes[c] for c in self.dataset.megacomplexes]
#          oscs = [self.model.oscillations[o] for cmplx in cmplxs
#                  for o in cmplx.oscillations]
#          oscs = list(set(oscs))
#          for osc in oscs:
#              compartment_order.append(osc.sin_compartment)
#              compartment_order.append(osc.cos_compartment)
#          return compartment_order
