# Introduction

Pyglotaran is a python library for global analysis of time-resolved spectroscopy data.
It is designed to provide a state of the art modeling toolbox to researchers, in a user-friendly manner.

Its features are:

- user-friendly modeling with a custom YAML (`*.yml`) based modeling language
- parameter optimization using variable projection and non-negative least-squares algorithms
- easy to extend modeling framework
- battle-hardened model and algorithms for fluorescence dynamics
- build upon and fully integrated in the standard Python science stack (NumPy,  SciPy, Jupyter)

## A Note To Glotaran Users

Although closely related and developed in the same lab, pyglotaran is not a
replacement for Glotaran - A GUI For TIMP. Pyglotaran only aims to provide the
modeling and optimization framework and algorithms. It is of course possible
to develop a new GUI which leverages the power of pyglotaran (contributions welcome).

The current 'user-interface' for pyglotaran is Jupyter Notebook. It is designed to
seamlessly integrate in this environment and be compatible with all major
visualization and data analysis tools in the scientific python environment.

If you are a non-technical user, you should give these tools a try, there are
numerous tutorials how to use them. You don't need to really learn to program.
If you can use e.g. Matlab or Mathematica, you can use Jupyter and Python.
