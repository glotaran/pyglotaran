# pyGloTarAn

pyGloTarAn is a python library for global and target analysis

[![latest release](https://pypip.in/version/pyglotaran/badge.svg)](https://pypi.org/project/pyglotaran/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyglotaran.svg)](https://anaconda.org/conda-forge/pyglotaran)
![Tests](https://github.com/glotaran/pyglotaran/workflows/Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyglotaran/badge/?version=latest)](https://pyglotaran.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/gh/glotaran/pyglotaran/branch/master/graph/badge.svg)](https://codecov.io/gh/glotaran/pyglotaran)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/glotaran/pyglotaran.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/glotaran/pyglotaran/alerts/)

## Warning

This project is still in a pre-alpha release stage and should only be used with care.

## Additional warning for scientists

The algorithms provided by this package still need to be validated and reviewed, pending the official release it should not be used in scientific publications.

## Temporary rename notice (!!!)

This package was previously released on [pypi](https://pypi.org/) under the name [glotaran](https://pypi.org/project/glotaran/) but was changed so as to not confuse it with the original Glotaran software, first published in the Journal of Statistical Software titeled [Glotaran: A Java-Based Graphical User Interface for the R Package TIMP](https://www.jstatsoft.org/article/view/v049i03) DOI: [10.18637/jss.v049.i03](https://www.jstatsoft.org/article/view/v049i03)

# Installation

Prerequisites:

- Python 3.6 or higher _(Python 2 is **not** supported)_
- On Windows only 64bit is supported

Note for Windows Users: The easiest way to get python for Windows is via [Anaconda](https://www.anaconda.com/)

## Stable Release

To install pyglotaran from [pypi](https://pypi.org/), run this command in your terminal:

```
$ pip install pyglotaran
```

If you want to install it via conda, you can run the following command:

```
$ conda install -c conda-forge pyglotaran
```

## From Source

```
$ git clone https://github.com/glotaran/pyglotaran.git
$ cd pyglotaran

$ pip install .
# To enforce python3 on systems where python2 is also installed
$ pip3 install .

```

_Note for Anaconda Users: Please make sure to update your distribution prior to install since some packages managed by Anaconda cannot be updated by `pip`._

# Mailinglist

[mailing-list](https://groups.google.com/forum/#!forum/glotaran)

## Credits

The credits can be found in the documentation
[authors section](https://pyglotaran.readthedocs.io/en/latest/authors.html)
