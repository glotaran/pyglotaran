# pyGloTarAn

pyGloTarAn is a Python library for Global and Target Analysis

[![latest release](https://pypip.in/version/pyglotaran/badge.svg)](https://pypi.org/project/pyglotaran/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyglotaran.svg)](https://anaconda.org/conda-forge/pyglotaran)
![Tests](https://github.com/glotaran/pyglotaran/workflows/Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyglotaran/badge/?version=latest)](https://pyglotaran.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/gh/glotaran/pyglotaran/branch/master/graph/badge.svg)](https://codecov.io/gh/glotaran/pyglotaran)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/glotaran/pyglotaran.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/glotaran/pyglotaran/alerts/)

## Installation

Prerequisites:

- Python 3.8 or 3.9 higher _(Python 2 is **not** supported)_
- On Windows only 64bit is supported

Note for Windows Users: The easiest way to get python for Windows is via [Anaconda](https://www.anaconda.com/)

### Stable Release

To install pyglotaran from [pypi](https://pypi.org/), run this command in your terminal:

```console
pip install pyglotaran
```

If you want to install it via conda, you can run the following command:

```shell
conda install -c conda-forge pyglotaran
```

### From Source

To install from source, e.g. for testing or development purposes, run these commands in your shell/terminal:

```shell
git clone https://github.com/glotaran/pyglotaran.git
cd pyglotaran
pip install .
```

_To enforce the installation within a Python3 environment on systems where Python2 is the default, instead of the last command use `pip3 install .`.

_Note for Anaconda Users: Please make sure to update your distribution prior to install since some packages managed by Anaconda cannot be updated by `pip`._

## Notice for scientists

Anyone using this package for serious work, in particular scientists and academic users, are cautioned, and treat it as any other instrument or tool that needs calibration or validation. The software comes with no warranties or guarantees as per the [LICENSE](LICENSE).

As of yet, the pyglotaran package has not yet been used in scientific publications (contribution welcome) but it has undergone some form of validation by cross-validating it against the TIM software package in use for several decades and described in [this publication](https://doi.org/10.1016/j.bbabio.2004.04.011) (DOI: [10.1016/j.bbabio.2004.04.011](https://doi.org/10.1016/j.bbabio.2004.04.011)). The examples used in this validation process can be obtained from the [pyglotaran_examples repository](https://github.com/glotaran/pyglotaran_examples).

## Glotaran legacy

The pyglotaran package derives its name from the glotaran software package (now called [glotaran-legacy](https://github.com/glotaran/glotaran-legacy)), first released in 2011 and described in a publication in the Journal of Statistical Software under the tile [Glotaran: A Java-Based Graphical User Interface for the R Package TIMP](https://www.jstatsoft.org/article/view/v049i03) ( DOI: [10.18637/jss.v049.i03](http://dx.doi.org/10.18637/jss.v049.i03) ).

## Mailing List

[mailing-list](https://groups.google.com/forum/#!forum/glotaran)

## Credits

The credits can be found in the documentation
[authors section](https://pyglotaran.readthedocs.io/en/latest/authors.html)
