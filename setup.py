from setuptools import setup, find_packages

install_requires = [
    'click>=7.0',
    'dask>=2.0',
    'dask[bag]>=2.0',
    'distributed>=2.0',
    'fsspec>=0.4.1',
    'lmfit>=0.9.13',
    'netCDF4>=1.5',
    'numba>=0.44',
    'numpy>=1.16',
    'pandas>=0.24',
    'partd>=0.3.10',  # Needed by dask
    'pyyaml>=5.1',
    'scipy>=1.3',
    'setuptools>=41.0',
    'xarray>=0.12',
]


with open("README.md", "r") as fh:
    long_description = fh.read()

entry_points = """
    [console_scripts]
    glotaran=glotaran.cli.main:glotaran

    [glotaran.plugins]
    kinetic_image_model = glotaran.builtin.models.kinetic_image
    kinetic_spectrum_model = glotaran.builtin.models.kinetic_spectrum
    doas_model = glotaran.builtin.models.doas

    ascii_file = glotaran.builtin.file_formats.ascii
    sdt_file = glotaran.builtin.file_formats.sdt
"""

setup(
    name="glotaran",
    version='0.0.12',
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
                    j.snellenburg@gmail.com,
                    i.h.m.van.stokkum@vu.nl """,
    license='GPLv3',
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points=entry_points,
    test_suite='glotaran',
    tests_require=['pytest'],
    zip_safe=True
)
