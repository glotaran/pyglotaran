from c_matrix_gaussian_irf import calculateCMultiGaussian


class CMatrixCython(object):
    def c_matrix(self, rates, times, centers, widths, scale):
        raise NotImplementedError

    def c_matrix_gaussian_irf(self, C, rates, times, centers, widths, scale):
        calculateCMultiGaussian(C, rates, times, centers, widths,
                                scale)
