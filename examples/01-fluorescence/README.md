# Example Case Study 1 - PS1 Fluorescence

Time-resolved florescence of a Photosystem 1 Spirulina Trimer excited at 400nm

## Contents

Used in the example:

- `data/data.ascii`: the dataset used in analysis
- [analysis.ipynb](analysis.ipynb): the notebook used for analysis
- [scheme.yaml](scheme.yaml): the analysis scheme
- [parameters.yaml](parameters.yaml): parameters used in the scheme

Supporting information:

- supporting_information: e.g. (partial) figures from the paper.
- expected_results: the expected results used for QA testing purposes

## Data

File: `data/data.ascii`:
- Format: `*.ascii` [time explicit format](https://glotaran.github.io/legacy/file_formats#time-explicit-format)
- Time dimension: n=923, min=-101, max=99.46
- Spectral dimension: n=49, min=626.1, max=788.7
- Datapoints: 45227

## Background Information

### Description

This example illustrates the global and target analysis of the time-resolved emission of a Photosystem I (PS-I) system following excitation at 400 nm. It loosely follows the analysis as described in the 2001 publication titled [Time-Resolved Fluorescence Emission Measurements of Photosystem I Particles of Various Cyanobacteria: A Unified Compartmental Model](https://www.cell.com/biophysj/fulltext/S0006-3495(01)75709-8). The sample in question is described as "Trimeric core of Spirulina platensis" in the referenced publication. Figure 4E of the publication (partially reproduced below) depicts the results of (global) analysis of the data, in the form of the "Decay Associated Spectra of fluorescence decay". Figure 5 (reproduced below) details the "Compartmental model describing the kinetics of different cyanobacterial PS-I core particles upon excitation at 400 nm".

Insert Figure 4E + title

Insert Figure 5 + title

## Citation

When using the data or schemes described in this example case study, please consider citing one of the following publication:

- Gobets, Bas et al. - Time-Resolved Fluorescence Emission Measurements of Photosystem I Particles of Various Cyanobacteria: A Unified Compartmental Model - Biophysical Journal, Volume 81, Issue 1, 407 - 424. DOI: [10.1016/S0006-3495(01)75709-8](https://doi.org/10.1016/S0006-3495(01)75709-8)
