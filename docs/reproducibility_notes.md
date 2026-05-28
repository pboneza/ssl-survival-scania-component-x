# Reproducibility Notes

This repository accompanies a master's thesis on semi-supervised survival modelling for reliability estimation using the SCANIA Component X dataset.

The repository is intended to support approximate reproducibility of the main experimental pipeline. The goal is to allow other researchers to inspect the implementation, reproduce the preprocessing steps, train the baseline models, train the proposed semi-supervised model, and obtain results that are broadly consistent with those reported in the thesis.

## Approximate Reproducibility

Exact numerical reproduction may not always be possible. Several parts of the experimental pipeline include stochastic components, such as:

- random train-validation splitting in some experimental stages;
- random truncation of vehicle histories;
- neural network weight initialization;
- contrastive learning augmentations;
- mini-batch ordering;
- early stopping behaviour;
- GPU-related nondeterminism.

Fixed random seeds are used where possible to reduce variation. However, small differences in results may still occur across machines, operating systems, Python versions, package versions, and GPU hardware.

## Data Availability

The SCANIA Component X dataset is not included in this repository. Users should obtain the dataset from the official source or through this link https://doi.org/10.5878/jvb5-d390 and place the raw files in the `data/raw/` directory. Only the training subset of the full dataset was used for these experiments as it is the only one with complete operational histories available.

The repository does not track raw data, processed data, model checkpoints, or generated intermediate files.

## Expected Workflow

The intended workflow is:

1. download the SCANIA Component X dataset;
2. place the required raw files in `data/raw/`;
3. install the required Python dependencies;
4. run the notebooks or scripts in numerical order;
5. compare the generated results with the thesis findings.

## Notes on Model Results

The main finding of the thesis was that the proposed semi-supervised survival modelling approach achieved better predictive performance than the evaluated classical survival baselines under the same truncation-based time-to-failure prediction setting.

Because of stochastic training and hardware differences, reproduced values should be interpreted in terms of overall trends and model rankings rather than exact numerical equality.