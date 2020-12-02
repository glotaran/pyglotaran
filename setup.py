from setuptools import find_packages
from setuptools import setup

install_requires = [
    "click>=7.0",
    "lmfit>=0.9.13",
    "netCDF4>=1.5.3",
    "numba>=0.48",
    'numpy>=1.17.3; sys_platform != "win32"',
    'numpy==1.19.3; sys_platform == "win32"',
    "pandas>=0.25.2",
    "pyyaml>=5.2",
    "scipy>=1.3.2",
    "sdtfile>=2020.8.3",
    "setuptools>=41.2",
    "xarray>=0.14",
]


with open("README.md") as fh:
    long_description = fh.read()

entry_points = """
    [console_scripts]
    glotaran=glotaran.cli.main:glotaran

    [glotaran.plugins]
    kinetic_image_model = glotaran.builtin.models.kinetic_image
    kinetic_spectrum_model = glotaran.builtin.models.kinetic_spectrum

    ascii_file = glotaran.builtin.file_formats.ascii
    sdt_file = glotaran.builtin.file_formats.sdt
"""

setup(
    name="pyglotaran",
    version="0.2.0",
    description="The Glotaran fitting engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/glotaran/pyglotaran",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    author="Joern Weissenborn, Joris Snellenburg, Ivo van Stokkum",
    author_email="""joern.weissenborn@gmail.com,
                    j.snellenburg@gmail.com,
                    i.h.m.van.stokkum@vu.nl """,
    license="GPLv3",
    project_urls={
        "GloTarAn Ecosystem": "http://glotaran.org",
        "Documentation": "https://glotaran.readthedocs.io",
        "Source": "https://github.com/glotaran/pyglotaran",
        "Tracker": "https://github.com/glotaran/pyglotaran/issues",
    },
    python_requires=">=3.8, <3.9",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points=entry_points,
    test_suite="glotaran",
    tests_require=["pytest"],
    zip_safe=True,
)
