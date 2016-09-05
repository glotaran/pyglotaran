from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[
    Extension("calculateC",
              ["calculateC.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              #define_macros=[("CYTHON_TRACE_NOGIL", "1")]
              ),
#    Extension("solve",
#              ["solve.pyx"],
#              include_dirs=[numpy.get_include()],
#              libraries=["m"],
#              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
#              extra_link_args=['-fopenmp']
#              #define_macros=[("CYTHON_TRACE_NOGIL", "1")]
#              )
]


setup(
    name = "large_example_irf_disp",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)