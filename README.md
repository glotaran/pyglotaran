# pyglotaran

pyglotaran is a Python library for Global and Target Analysis

[![latest release](https://pypip.in/version/pyglotaran/badge.svg)](https://pypi.org/project/pyglotaran/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyglotaran.svg)](https://anaconda.org/conda-forge/pyglotaran)
![Tests](https://github.com/glotaran/pyglotaran/workflows/Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyglotaran/badge/?version=latest)](https://pyglotaran.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/glotaran/pyglotaran.git/main?urlpath=lab%2Ftree%2Fdocs%2Fsource%2Fnotebooks)
[![Coverage Status](https://codecov.io/gh/glotaran/pyglotaran/branch/master/graph/badge.svg)](https://codecov.io/gh/glotaran/pyglotaran)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/glotaran/pyglotaran.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/glotaran/pyglotaran/alerts/)

**Warning**: This is an _early access_ release, please refer to the [usage notice](#usage-notice) down below prior to committing to use pyglotaran to avoid surprises down the line.

## Installation

Prerequisites:

- Python 3.8 or 3.9
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

\_To enforce the installation within a Python3 environment on systems where Python2 is the default, instead of the last command use `pip3 install .`.

_Note for Anaconda Users: Please make sure to update your distribution prior to install since some packages managed by Anaconda cannot be updated by `pip`._

## Usage notice

This software package is made available as an early access release, to the advantage of the (scientific) community who wishes to make use of it before it has fully matured, but without any warranties.

Anyone using this package for serious work - scientists and academic users in particular - are cautioned, and treat it as any other instrument or tool that requires calibration or validation. Also be prepared for some refactoring of models or analysis specifications, sometime down the line.

As of yet, the pyglotaran package has not yet been used in any peer-reviewed scientific publications (contribution welcome), but it has been partially cross-validated against comparable software, such as the [R-package TIMP](https://dx.doi.org/10.18637/jss.v018.i03), and the TIM software described in [this publication](https://doi.org/10.1016/j.bbabio.2004.04.011) ([DOI: 10.1016/j.bbabio.2004.04.011](https://doi.org/10.1016/j.bbabio.2004.04.011)).

The examples used in this validation process can be obtained from the [pyglotaran-examples repository](https://github.com/glotaran/pyglotaran-examples).

## Glotaran legacy

The pyglotaran package derives its name from the Glotaran software package (now called [glotaran-legacy](https://github.com/glotaran/glotaran-legacy)), first released in 2011 and described in a highly-cited publication in the Journal of Statistical Software, under the title [Glotaran: A Java-Based Graphical User Interface for the R Package TIMP](https://www.jstatsoft.org/article/view/v049i03) ([DOI: 10.18637/jss.v049.i03](https://dx.doi.org/10.18637/jss.v049.i03)).

The [pyglotaran](https://github.com/glotaran/pyglotaran) software can be considered the spiritual successor of the [glotaran-legacy](https://github.com/glotaran/glotaran-legacy) software and has the backing of many of its original creators.

## The future of global and target analysis

Eventually, and hopefully sooner than later, a (scientific) publication about the architecture and the design of the software package will appear, detailing the flexibility of the software and showing why we believe this software package is the future of global and target analysis.

Until then, enjoy your glimpse into the future.

## Mailing List

[mailing-list](https://groups.google.com/forum/#!forum/glotaran)

## Credits

The credits can be found in the documentation
[authors section](https://pyglotaran.readthedocs.io/en/latest/authors.html)
