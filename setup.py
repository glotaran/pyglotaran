from setuptools import find_packages
from setuptools import setup

install_requires = [
    "asteval>=0.9.21",
    "click>=7.0",
    "netCDF4>=1.5.3",
    "numba>=0.48",
    "numpy>=1.19.5,<1.20.0",
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
    version="0.3.0",
    description="The Glotaran fitting engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/glotaran/pyglotaran",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
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
    license="LGPLv3",
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
