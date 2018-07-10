import os
import sys

# TODO: bootstrap numpy ->
# https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
import numpy
import scipy
# TODO: include generated c and include switches if cython is not available ->
# https://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
from Cython.Distutils import build_ext
from setuptools import setup, Command
from setuptools.extension import Extension


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.pyd ./*.tgz ./*.egg-info')

# TODO: 'win32' ok, else=linux/mac, what about 'win-amd64' and 'win-ia64'?
if sys.platform == 'win32':
    ext_modules = [
        Extension("c_matrix",
                  ["glotaran/models/spectral_temporal/c_matrix_cython/c_matrix.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include()],
                  extra_compile_args=["-O3", "-ffast-math", "-march=native",
                                      "-fopenmp"],
                  extra_link_args=['-fopenmp']),
        Extension("c_matrix_gaussian_irf",
                  ["glotaran/models/spectral_temporal/c_matrix_cython/c_matrix_gaussian_irf.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include()],
                  extra_compile_args=["-O3", "-ffast-math", "-march=native",
                                      "-fopenmp"],
                  extra_link_args=['-fopenmp'])
                  ]
else:
    ext_modules = [
        Extension("c_matrix",
                  ["glotaran/models/spectral_temporal/c_matrix_cython/c_matrix.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include()],
                  libraries=["m"],
                  extra_compile_args=["-O3", "-ffast-math", "-march=native",
                                      "-fopenmp"],
                  extra_link_args=['-fopenmp']),
        Extension("c_matrix_gaussian_irf",
                  ["glotaran/models/spectral_temporal/c_matrix_cython/c_matrix_gaussian_irf.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include()],
                  libraries=["m"],
                  extra_compile_args=["-O3", "-ffast-math", "-march=native",
                                      "-fopenmp"],
                  extra_link_args=['-fopenmp'])
                  ]


setup(
    name="glotaran",
    version="0.0.1",
    description='The Glotaran fitting engine.',
    url='http://glotaran.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Cython',
        'Topic :: Scientific'
    ],
    author='Joris Snellenburg, Joern Weissenborn, Ivo van Stokkum',
    author_email="""j.snellenburg@gmail.com,
                    joern.weissenborn@gmail.com,
                    i.h.m.van.stokkum@vu.nl """,
    license='GPLv3',
    packages=[
              'glotaran',
              'glotaran.dataio',
              'glotaran.plotting',
              'glotaran.model',
              'glotaran.fitmodel',
              'glotaran.models.spectral_temporal',
              'glotaran.models.spectral_temporal.c_matrix_cython',
              'glotaran.specification_parser'
              ],
    install_requires=[
        'numpy>=1.9.1',
        'scipy>=1.0.0',
        'lmfit>=0.9.7',
        'pandas>=0.23.1',
        'pyyaml',
        'matplotlib',  # dependency introduced by glotaran.plotting
        'natsort',  # dependency introduced by glotaran.data.io.chlorospec_format
        'lmfit-varpro>=0.1.0'
    ],
	dependency_links=['https://github.com/glotaran/lmfit-varpro/tarball/master#egg=lmfit-varpro-0.1.0'],
    cmdclass={"build_ext": build_ext, 'clean': CleanCommand},
    ext_modules=ext_modules,
    test_suite='nose.collector',
    tests_require=['nose'],
)

#    package_data={'glotaran.models.kinetic.c_matrix_opencl':
#                  ['*.cl']},
