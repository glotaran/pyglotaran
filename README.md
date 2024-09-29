# pyglotaran

<img align="right" width="100" height="100" src="https://raw.githubusercontent.com/glotaran/pyglotaran/main/docs/source/images/pyglotaran_logo_transparent.svg">

[![PyPI version](https://badge.fury.io/py/pyglotaran.svg)](https://badge.fury.io/py/pyglotaran)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyglotaran.svg)](https://anaconda.org/conda-forge/pyglotaran)
![Tests](https://github.com/glotaran/pyglotaran/workflows/Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyglotaran/badge/?version=latest)](https://pyglotaran.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/gh/glotaran/pyglotaran/branch/master/graph/badge.svg)](https://codecov.io/gh/glotaran/pyglotaran)
[![CodeQL](https://github.com/glotaran/pyglotaran/actions/workflows/codeql.yml/badge.svg)](https://github.com/glotaran/pyglotaran/actions/workflows/codeql.yml)
[![Discord](https://img.shields.io/discord/883443835135475753.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/KfnEYRSTJx)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4534043.svg)](https://doi.org/10.5281/zenodo.4534043)

A framework written in Python for global and target analysis; commonly used for the analysis of time-resolved spectroscopy measurements in the study of energy transfer pathways in photosynthesis, or the characterization of energy transfer (in-)efficiencies in photovoltaics.

## The Future of Global and Target Analysis

A scientific publication titled "pyglotaran: a lego-like Python framework for global and target analysis of time-resolved spectra" (DOI: [10.1007/s43630-023-00460-y](https://doi.org/10.1007/s43630-023-00460-y)) covers aspects of the architecture and the design of the software while illustrating its flexibility as an analysis tool through some exciting case studies. This publication, along with other [pyglotaran-publications](https://github.com/glotaran/pyglotaran-publications) demonstrates why we believe this framework is the future of global and target analysis.

## Usage of pyglotaran

If you just want to get started analyzing your data, jump straight to the [getting started](https://pyglotaran.readthedocs.io/en/latest/notebooks/getting_started/getting_started.html) guide that is part of our [documentation](https://pyglotaran.readthedocs.io/en/latest/).

Afterwards, you may also want to have a look at our comprehensive collection of real-world examples in the [pyglotaran-examples](https://github.com/glotaran/pyglotaran-examples) repository. There most like is an example that aligns well with your use case.

For state of the art usage of this framework you may want to check out the list of [pyglotaran-publications](https://github.com/glotaran/pyglotaran-publications).

### Example analysis results

Results of a target analysis of Photosystem I (see [study_florescence](https://github.com/glotaran/pyglotaran-examples/tree/main/pyglotaran_examples/study_fluorescence) in the [examples](https://github.com/glotaran/pyglotaran-examples/tree/main/pyglotaran_examples).)

![overview plot of the fluorescence case study from the pyglotaran-examples](docs/source/images/png/plot_overview_study_fluorescence.png)
<sub>Plot created with the `plot_overview` function of the [pyglotaran-extras](https://github.com/glotaran/pyglotaran-extras) package.</sub>

<sub>**Notice**: This is (still) an _early access_ release, please refer to the [usage notice](NOTICE.md#usage-notice) prior to committing to use pyglotaran to avoid surprises down the line.</sub>

## Glotaran legacy

The pyglotaran package derives its name from the Glotaran software package (now called [glotaran-legacy](https://github.com/glotaran/glotaran-legacy)), first released in 2011 and described in a highly-cited publication in the Journal of Statistical Software, under the title [Glotaran: A Java-Based Graphical User Interface for the R Package TIMP](https://www.jstatsoft.org/article/view/v049i03) ([DOI: 10.18637/jss.v049.i03](https://dx.doi.org/10.18637/jss.v049.i03)).

The [pyglotaran](https://github.com/glotaran/pyglotaran) framework can be considered the spiritual successor of the [glotaran-legacy](https://github.com/glotaran/glotaran-legacy) software and has the backing of many of its original creators.

## Community Support

For questions / suggestion please reach out to us via:

1. [GitHub issues](https://github.com/glotaran/pyglotaran/issues)
2. [Discord](https://discord.gg/KfnEYRSTJx)
3. [Google-Groups mailing list](https://groups.google.com/forum/#!forum/glotaran)

## Credits

The credits can be found in the documentation
[authors section](https://pyglotaran.readthedocs.io/en/latest/authors.html)
