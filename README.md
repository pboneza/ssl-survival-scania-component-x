# Semi-Supervised Survival Modelling in Predictive Maintenance using the SCANIA Component X Dataset as a case study

This repository contains the code and experimental pipeline developed for a master's thesis on semi-supervised learning for survival modelling in predictive maintenance.

The study investigates whether semi-supervised representation learning can improve time-to-failure prediction under heavy right censoring and irregularly sampled multivariate time series data. The experiments are conducted using a case study of the SCANIA Component X dataset.

The repository is intended to support approximate reproducibility of the thesis experiments and to provide supplementary implementation material for future research on predictive maintenance, survival analysis, and semi-supervised learning.

## Research Objective

The main objective of this work is to investigate whether semi-supervised representation learning can improve performance in survival-based reliability estimation when only a limited number of failure events are available.

More specifically, the study evaluates whether representations learned from irregularly sampled operational histories can improve time-to-failure prediction compared with established, purely supervised survival models.

## Dataset

The experiments use the SCANIA Component X dataset, a real-world multivariate time series dataset for predictive maintenance.

The dataset includes:

- operational readouts from vehicles;
- time-to-event information;
- vehicle specification data.

The dataset is not included in this repository. Users should download it from the official source and place the raw files in:

```text
data/raw/

## Reproducibility Notes

Additional notes on approximate reproducibility are provided in:

```text
docs/reproducibility_notes.md