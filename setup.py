# This is just a placeholder setup.py to claim the glotaran name on PyPI
# It is not meant to be usuable in any way as of yet.
import os
import shutil
import sys

from setuptools import find_packages, Command

setup_requires = [
    'numpy>=1.15.1',
    'scipy>=1.2.0',
    'Cython>=0.28.5',
    'setuptools>=40.2.0'
]
install_requires = [
    'lmfit>=0.9.11',
    'pandas>=0.23.4',
    'pyyaml>=3.0,<=5.0',
    'xarray>=0.11.2',
    'natsort>=5.3.3',  # dependency introduced by glotaran.dataio.chlorospec_format
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
    from numpy.distutils.core import setup, Extension

    ext_modules = [
        Extension(name="kinetic_matrix_no_irf",
                  sources=["glotaran/models/spectral_temporal/kinetic_matrix_no_irf.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include(),
                                "glotaran/models/spectral_temporal"]),
        Extension("kinetic_matrix_gaussian_irf",
                  ["glotaran/models/spectral_temporal/erfce.c",
                   "glotaran/models/spectral_temporal/kinetic_matrix_gaussian_irf.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include(),
                                "glotaran/models/spectral_temporal"]),
        Extension(name="doas_matrix_faddeva",
                  sources=["glotaran/models/doas/doas_matrix_faddeva.pyx"],
                  include_dirs=[numpy.get_include(), scipy.get_include(),
                                "glotaran/models/doas"]),
        Extension(name="scalTOMS680",
                  sources=["glotaran/models/doas/scalTOMS680.pyf",
                           "glotaran/models/doas/scalTOMS680.f"],
                  include_dirs=[numpy.get_include(), scipy.get_include(),
                                "glotaran/models/doas"]),
        ]

except ImportError:
    raise ImportError(f"To install glotaran you need to have following packages installed:\n"
                      f"{setup_requires[0]}\n"
                      f"{setup_requires[1]}\n"
                      f"{setup_requires[2]}\n"
                      f"You can install them by running:\n"
                      f"`pip install '{setup_requires[0]}' '{setup_requires[1]}' "
                      f"'{setup_requires[2]}'`")


# Patch for np.distutil which has broken cython support.
# See https://stackoverflow.com/questions/37178055/attributeerror-list-object-has-no-attribute-rfind-using-petsc4py # noqa e501
from numpy.distutils.misc_util import appendpath  # noqa402
from numpy.distutils import log  # noqa402
from numpy.distutils.command import build_src # noqa402
from os.path import join as pjoin, dirname  # noqa402
from distutils.dep_util import newer_group  # noqa402
from distutils.errors import DistutilsError  # noqa402

import Cython.Compiler.Main  # noqa402
build_src.Pyrex = Cython
build_src.have_pyrex = True


def have_pyrex():
    import sys  # noqa402
    try:
        import Cython.Compiler.Main
        sys.modules['Pyrex'] = Cython
        sys.modules['Pyrex.Compiler'] = Cython.Compiler
        sys.modules['Pyrex.Compiler.Main'] = Cython.Compiler.Main
        return True
    except ImportError:
        return False


build_src.have_pyrex = have_pyrex


def generate_a_pyrex_source(self, base, ext_name, source, extension):
    ''' Monkey patch for numpy build_src.build_src method
    Uses Cython instead of Pyrex.
    Assumes Cython is present
    '''
    if self.inplace:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = pjoin(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    if self.force or newer_group(depends, target_file, 'newer'):
        import Cython.Compiler.Main
        log.info(f"cythonc:> {target_file}")
        self.mkpath(target_dir)
        options = Cython.Compiler.Main.CompilationOptions(
            defaults=Cython.Compiler.Main.default_options,
            include_path=extension.include_dirs,
            output_file=target_file)
        cython_result = Cython.Compiler.Main.compile(source, options=options)
        if cython_result.num_errors != 0:
            raise DistutilsError(
                f"{cython_result.num_errors} errors while compiling {source} with Cython")
    return target_file


build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source
########################
# END additionnal code #
########################


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
    author='Joern Weissenborn, Joris Snellenburg, Ivo van Stokkum',
    author_email="""joern.weissenborn@gmail.com,
                    j.snellenburg@gmail.com
                    i.h.m.van.stokkum@vu.nl """,
    license='GPLv3',
    python_requires=">=3.6",
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=setup_requires+install_requires,
    #  cmdclass={'clean': CleanCommand},
    ext_modules=ext_modules,
    test_suite='glotaran',
    tests_require=['pytest'],
    zip_safe=False
)

#    package_data={'glotaran.models.kinetic.c_matrix_opencl':
#                  ['*.cl']},
