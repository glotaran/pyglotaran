import pyopencl as cl
import numpy as np
import os

KERNEL_FILE = os.path.join(os.path.dirname(__file__), 'kernel_c_matrix_irf.cl')

mf = cl.mem_flags


class CMatrixOpenCL(object):

    def __init__(self):
        self._ctx = cl.create_some_context()
        self._queue = cl.CommandQueue(self._ctx)
        self._init_kernel()

    def _init_kernel(self):
        with open(KERNEL_FILE) as f:
            self._kernel = cl.Program(self._ctx, f.read()).build()

    def c_matrix(self, rates, times, centers, widths, scale):
        raise NotImplementedError

    def c_matrix_gaussian_irf(self, C, rates, times, centers, widths, scale):
        result = cl.Buffer(self._ctx, mf.WRITE_ONLY, C.nbytes)

        rates_dev = cl.Buffer(self._ctx,
                              mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=rates)
        
        times_dev = cl.Buffer(self._ctx,
                              mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=times)
        
        center_dev = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=centers)
        width_dev = cl.Buffer(self._ctx,
                              mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=widths)

        self._kernel.c_matrix_irf(self._queue, C.shape, None,
                                  np.int32(rates.shape[0]),
                                  np.int32(times.shape[0]),
                                  np.int32(centers.shape[1]),
                                  np.float64(1),
                                  rates_dev,
                                  times_dev,
                                  center_dev,
                                  width_dev,
                                  result)
        
        #  res_np = np.empty_like(a_np)
        cl.enqueue_copy(self._queue, C, result)
