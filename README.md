# pyGloTarAn

pyGloTarAn is a python library for global and target analysis

[![latest release](https://pypip.in/version/pyglotaran/badge.svg)](https://pypi.org/project/pyglotaran/)
![Tests](https://github.com/glotaran/pyglotaran/workflows/Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyglotaran/badge/?version=latest)](https://pyglotaran.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/gh/glotaran/pyglotaran/branch/master/graph/badge.svg)](https://codecov.io/gh/glotaran/pyglotaran)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=glotaran/pyglotaran)](https://dependabot.com)

## Warning

This project is still in a pre-alpha release stage and should only be used with care.

## Additional warning for scientists

The algorithms provided by this package still need to be validated and reviewed, pending the official release it should not be used in scientific publications.

## Temporary rename notice (!!!)

This package was previously released on [pypi](https://pypi.org/) under the name [glotaran](https://pypi.org/project/glotaran/) but was changed so as to not confuse it with the original Glotaran software, first published in the Journal of Statistical Software titeled [Glotaran: A Java-Based Graphical User Interface for the R Package TIMP](https://www.jstatsoft.org/article/view/v049i03) DOI: [10.18637/jss.v049.i03](https://www.jstatsoft.org/article/view/v049i03)


# Installation

## From Source

Prerequisites:

- Python 3.6 or higher _(Python 2 is **not** supported)_
- On Windows only 64bit is supported

Note for Windows Users: The easiest way to get python for Windows is via [Anaconda](https://www.anaconda.com/)

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

The credits can be found in the documentations
[credits section](https://pyglotaran.readthedocs.io/en/latest/credits.html)
