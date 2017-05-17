from c_matrix_gaussian_irf import calculateCSingleGaussian
from c_matrix import calculateC


class CMatrixCython(object):
    def c_matrix(self, C, idxs, rates, times, scale):
        return calculateC(C, idxs, rates, times, scale)

    def c_matrix_gaussian_irf(self, C, idxs, rates, times, centers, widths,
                              scale, backsweep, backsweep_period):
        calculateCSingleGaussian(C, idxs, rates, times, centers, widths,
                                 scale, backsweep, backsweep_period)
