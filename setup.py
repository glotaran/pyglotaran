# This is just a placeholder setup.py to claim the glotaran name on PyPI
# It is not meant to be usuable in any way as of yet.
import os
import shutil
import sys

from setuptools import setup, find_packages, Command
from setuptools.extension import Extension

setup_requires = [
    'numpy>=1.15.1',
    'scipy>=1.1.0',
    'Cython>=0.28.5',
    'setuptools>=40.2.0'
]
install_requires = [
    'lmfit>=0.9.11',
    'pandas>=0.23.4',
    'pyyaml>=3.13',
    'matplotlib>=2.2.3',  # dependency introduced by glotaran.plotting
    'natsort>=5.3.3',  # dependency introduced by glotaran.dataio.chlorospec_format
    'lmfit-varpro>=0.0.5',
    'typing_inspect>=0.3.1',
]

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
        Extension("kinetic_matrix_no_irf",
                  ["glotaran/models/spectral_temporal/kinetic_matrix_no_irf.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include(),
                                "glotaran/models/spectral_temporal"]),
        Extension("kinetic_matrix_gaussian_irf",
                  ["glotaran/models/spectral_temporal/erfce.c",
                   "glotaran/models/spectral_temporal/kinetic_matrix_gaussian_irf.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include(),
                                "glotaran/models/spectral_temporal"]),
        #  Extension("c_matrix_damped_oscillation",
        #            ["glotaran/models/damped_oscillation/c_matrix_damped_oscillations.pyx"],
        #            include_dirs=[numpy.get_include(), scipy.get_include(),
        #                          "glotaran/models/damped_oscillation"]),
                  ]

except ImportError:
    raise ImportError(f"To install glotaran you need to have following packages installed:\n"
                      f"{setup_requires[0]}\n"
                      f"{setup_requires[1]}\n"
                      f"{setup_requires[2]}\n"
                      f"You can install them by running:\n"
                      f"`pip install '{setup_requires[0]}' '{setup_requires[1]}' '{setup_requires[2]}'`")

# backport of dataclases only needed for python 3.6

if sys.version_info.major == 3 and sys.version_info.minor == 6:
    install_requires.append('dataclasses>=0.6')

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="glotaran",
    version='0.0.10',
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
