# Sequential kernelized independence testing

Code for the paper: [Sequential Kernelized Independence Testing](https://arxiv.org/abs/2212.07383).

## Contents

- gaussian_example.ipynb is an illustation of the SKIT deployed on Gaussian linear model.

- Folder "notebooks for plots in the paper" contains the notebooks for reproducing plots from the paper

- Folder "utils" contains
  - testing.py (SKIT, Batch HSIC, Median Heuristic Implementation)
  - payoff_fns.py (implementation of different payoffs)
  - decomp.py (Incomplete Cholesky decomposition for COCO/KCC payoffs)
  - data_gen.py (some functions for generating synthetic data)
