from c_matrix_gaussian_irf import calculateCSingleGaussian
from c_matrix import calculateC


class CMatrixCython(object):
    def c_matrix(self, C, rates, times):
        return calculateC(C, rates, times)

    def c_matrix_gaussian_irf(self, C, rates, times, centers, widths, scale):
        calculateCSingleGaussian(C, rates, times, centers, widths,
                                 scale)
