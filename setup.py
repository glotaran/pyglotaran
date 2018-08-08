# This is just a placeholder setup.py to claim the glotaran name on PyPI
# It is not meant to be usuable in any way as of yet.
import os
import shutil

from setuptools import setup, find_packages, Command
from setuptools.extension import Extension


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_tree = [
            os.path.abspath("./build"),
            os.path.abspath("./dist"),
        ]
        for root, dirs, filenames in os.walk(os.path.abspath(".")):
            for dir in dirs:
                if dir.endswith(("__pycache__", ".egg-info", ".pytest_cache")) and \
                        ".tox" not in root:

                    self._clean_tree.append(os.path.join(root, dir))

            for filename in filenames:
                if filename.endswith((".pyc", ".pyd", ".tgz", ".egg-info")) and \
                        ".tox" not in root and root not in self._clean_tree:

                    self._clean_tree.append(os.path.join(root, filename))

    def finalize_options(self):
        pass

    def run(self):
        for clean_path in self._clean_tree:
            try:
                shutil.rmtree(clean_path)
            except Exception:
                pass


try:
    import numpy
    import scipy
    ext_modules = [
        Extension("c_matrix",
                  ["glotaran/models/spectral_temporal/c_matrix_cython/c_matrix.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include(),
                                "glotaran/models/spectral_temporal/c_matrix_cython"]),
        Extension("c_matrix_gaussian_irf",
                  ["glotaran/models/spectral_temporal/c_matrix_cython/c_matrix_gaussian_irf.pyx",
                   "glotaran/models/spectral_temporal/c_matrix_cython/erfce.c"],
                  include_dirs=[numpy.get_include(), scipy.get_include(),
                                "glotaran/models/spectral_temporal/c_matrix_cython"]),
        #  Extension("c_matrix_damped_oscillation",
        #            ["glotaran/models/damped_oscillation/c_matrix_damped_oscillations.pyx"],
        #            include_dirs=[numpy.get_include(), scipy.get_include(),
        #                          "glotaran/models/damped_oscillation"]),
                  ]

except ImportError:
    raise ImportError("To install glotaran you need to have following packages installed:\n"
                      "numpy>=1.9.1\n"
                      "scipy>=1.0.0\n"
                      "Cython>=0.28.3\n"
                      "You can install them by running:\n"
                      "`pip install 'numpy>=1.9.1' 'scipy>=1.0.0' 'Cython>=0.28.3'`")

setup_requires = [
    'numpy>=1.9.1',
    'scipy>=1.0.0',
    'Cython>=0.28.3',
    'setuptools>=39.2.0'
]
install_requires = [
    'lmfit>=0.9.7',
    'pandas>=0.23.1',
    'pyyaml',
    'matplotlib',  # dependency introduced by glotaran.plotting
    'natsort',  # dependency introduced by glotaran.data.io.chlorospec_format
    'lmfit-varpro>=0.0.2'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="glotaran",
    version='0.0.7',
    description='The Glotaran fitting engine.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://glotaran.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    author='Joris Snellenburg, Joern Weissenborn, Ivo van Stokkum',
    author_email="""j.snellenburg@gmail.com,
                    joern.weissenborn@gmail.com,
                    i.h.m.van.stokkum@vu.nl """,
    license='GPLv3',
    python_requires=">=3.6",
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=setup_requires+install_requires,
    cmdclass={'clean': CleanCommand},
    ext_modules=ext_modules,
    test_suite='glotaran',
    tests_require=['pytest'],
    zip_safe=False
)

#    package_data={'glotaran.models.kinetic.c_matrix_opencl':
#                  ['*.cl']},
