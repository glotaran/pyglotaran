from c_matrix_gaussian_irf import calculateCSingleGaussian
from c_matrix import calculateC


class CMatrixCython(object):
    def c_matrix(self, C, idxs, rates, times):
        return calculateC(C, idxs, rates, times)

    def c_matrix_gaussian_irf(self, C, idxs, rates, times, centers, widths,
                              scale):
        calculateCSingleGaussian(C, idxs, rates, times, centers, widths,
                                 scale)
