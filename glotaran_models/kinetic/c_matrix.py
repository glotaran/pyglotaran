class CMatrixBackend(object):
    def c_matrix(self, rates, times, centers, widths, scale):
        raise NotImplementedError

    def c_matrix_gaussian_irf(self, rates, times, centers, widths, scale):
        raise NotImplementedError
