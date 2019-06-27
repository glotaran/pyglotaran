import os
import shutil

from setuptools import setup, find_packages, Command
from setuptools.extension import Extension

setup_requires = [
    'numpy>=1.16.2',
    'scipy>=1.2.0',
    'setuptools>=40.2.0'
]
install_requires = [
    'click>=7.0',
    'dask>=1.2',
    'distributed>=1.28',
    'lmfit>=0.9.12',
    'natsort>=5.3.3',  # dependency introduced by glotaran.dataio.chlorospec_format
    'netCDF4>=1.4.2',
    'numba>=0.44',
    'pandas>=0.23.4',
    'pyyaml>=3.0,<=5.0',
    'xarray>=0.12.1',
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

except ImportError:
    raise ImportError(f"To install glotaran you need to have following packages installed:\n"
                      f"{setup_requires[0]}\n"
                      f"{setup_requires[1]}\n"
                      f"{setup_requires[2]}\n"
                      f"You can install them by running:\n"
                      f"`pip install '{setup_requires[0]}' '{setup_requires[1]}' "
                      f"'{setup_requires[2]}'`")


with open("README.md", "r") as fh:
    long_description = fh.read()

entry_points = """
    [console_scripts]
    glotaran=glotaran.cli.main:glotaran

    [glotaran.plugins]
    kinetic_image_model = glotaran.plugins.builtin.models.kinetic_image
    kinetic_spectrum_model = glotaran.plugins.builtin.models.kinetic_spectrum
    doas_model = glotaran.plugins.builtin.models.doas
"""

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
    entry_points=entry_points,
    test_suite='glotaran',
    tests_require=['pytest'],
    zip_safe=False
)
