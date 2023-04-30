# Sequential kernelized independence testing

Code for the paper: [Sequential Kernelized Independence Testing](https://arxiv.org/abs/2212.07383).

Sequential Independence Tests process data online and control the type-1 error despite continuously monitoring data.

This [notebook](/gaussian_example.ipynb) contains an illustation of SKIT deployed on Gaussian linear model.

## Contents

- Folder "utils" contains
  - testing.py (SKIT, Batch HSIC, Median Heuristic Implementation).
  - payoff_fns.py (Different payoffs functions).
  - decomp.py (Incomplete Cholesky decomposition for COCO/KCC payoffs).
  - data_gen.py (Some functions for generating synthetic data).
