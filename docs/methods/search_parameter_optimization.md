# Optimization and Calibration
In peptide centric DIA search, calibration of the library and optimization of search parameters is required to maximize the number of confident identifications. AlphaDIA performs both calibration and optimization iteratively. Calibration removes the systematic deviation of observed and library values to account for technical variation from the LC or MS instrument. Optimization reduces the search space to improve the confidence in identifications and to accelerate search.
:::{note}
Calibration and optimization are different but both connected to transfer learning. In [transfer learning](./transfer-learning.md) the residual (non-systematic) variation is learned and thereby reduced. This usually leads to better performance if used with optimization and calibration.
:::

## Overview of optimization
AlphaDIA can perform optimization for the following parameters, `target_ms1_tolerance`, `target_ms2_tolerance`, `target_mobility_tolerance` and `target_rt_tolerance`. There are two optimization strategies, targeted and automatic.

### Targeted optimization
The search space is progressively narrowed until a target tolerance is reached for a given parameter.

To activate targeted optimization for example for fragment m/z tolerance, set `target_ms2_tolerance` to `10` for using a target tolerance of 10 ppm.
For retention time, the target value can be either set as an absolute value in seconds or as a fraction of the total retention time range.

For example, setting `target_rt_tolerance` to `300` will result in a target tolerance of 300 seconds, while setting it to `0.3` will use 30% of the gradient length as the target tolerance.

### Automatic optimization
In automatic optimization the search space is reduced until an optimal value is detected. This optimization is curently performed for every raw file individually. The results of the optimization can be found in the [stats.tsv](<project:../methods/output-format.md>) file in the output directory.

To activate automatic optimization for a certain quantity, set the respective target to `0.0`, for example set `target_rt_tolerance=0.0` for automatic retention time optimization.

:::{tip}
We recommend to always use automatic optimization for retention time and ion mobility as set by default. For automatic optimization of mass tolerances we recommend using optimization in the first pass and then using the optimized values in the second pass.
:::

## Optimization and Calibration Algorithm
### Overall process
AlphaDIA performs iterative optimization and calibration of retention time, ion mobility, precursor m/z and fragment m/z parameters as illustrated below.

<img src="../_static/images/methods_optimization.png" width="100%" height="auto">

Optimization can be performed in either a targeted or automatic manner. In targeted optimization, the search space is progressively narrowed until a target tolerance is reached for a given parameter. In automatic optimization, the search space is progressively narrowed until an internal algorithm detects that further narrowing will reduce the confident identification of precursors (either by directly assessing the proportion of the library which has been detected or using a surrogate metric, such as the mean isotope intensity correlation for precursor m/z tolerance), at which point the optimal value is selected for search. It is possible to use targeted optimization for some parameters and automatic optimization for others.

AlphaDIA iteratively performs calibration and optimization based on a subset of the spectral library used for search. The size of this subset is adjusted according to an exponential batch plan to balance accuracy and efficiency. A defined number of precursors, set by the ``optimization_lock_target`` (default: 200), need to be identified at 1% FDR before calibration and optimization are performed. If fewer precursors than the target number are identified using a given step of the batch plan, AlphaDIA will search for precursors from the next step of the batch plan in addition to those already searched. If more precursors than the target number are identified, AlphaDIA will check if any previous step of the batch plan is also likely to yield at least the target number, in which case it will use the smallest such step of the batch plan for the next iteration of calibration and optimization. In this way, AlphaDIA ensures that calibration is always performed on sufficient precursors to be reliable, while calibrating on the smallest-possible subset of the library to maximize efficiency.

The process of choosing the best batch step for calibration and optimization is illustrated in the figure below, which shows optimization and calibration over seven iterations; the batch size is increased in the first iteration until sufficient precursors are detected, and subsequently reduced when the proportion is sufficiently high that a previous step should reach the target as well; if, however, fluctuations in the number of identifications mean that not enough precursors are actually identified, the next step in the batch plan will be searched as well to ensure that calibration is always performed on at least the target number of precursors.

<img src="../_static/images/methods_optimization_calibration.png" width="100%" height="auto">

### Calibration
If enough confident target precursors have been detected, they are calibrated to the observed values using locally estimated scatterplot smoothing (LOESS) regression. For calibration of fragment m/z values, a fixed number of up to 5000 of the best fragments according to their  XIC correlation are used. For precursor calibration all precursors passing 1% FDR are used. Calibration is performed prior to every optimization.

### Optimization
For optimizing the search space, tolerances like retention time, ion mobility and m/z ratios need to be reduced. The goal is to cover the expected spectrum space but reduce it as much as possible to accelerate search and gain statistical power. Search starts with initial tolerances as defined in `search_initial`.  For targeted optimization, the 95% deviation after calibration is adopted as the new tolerance until the target tolerances defined in the `search` section are reached. For automatic optimization, the 99% deviation plus 10% of the absolute value of the tolerance is adopted as the new tolerance, and search continues until parameter-specific convergence rules are met.

The optimization is finished as soon as the minimum number of steps `min_steps` has passed and all tolerances have either 1. reached the target tolerances defined in `search` if using targeted optimization, or 2. have converged if using automatic optimization.

## Configuring calibration and optimization
By default, alphaDIA performs targeted optimization of precursor m/z, fragment m/z, and automatic optimization of retention time and ion mobility using the settings below.

```yaml
search:
  # Number of peak groups identified in the convolution score to classify with target decoy competition
  target_num_candidates: 2

  # Targeted optimization of precursor m/z tolerance.
  # Use absolute values in ppm (e.g. 15ppm) or set to 0 for automatic optimization.
  target_ms1_tolerance: 5

  # Targeted optimization of fragment m/z tolerance.
  # Use absolute values in ppm (e.g. 15ppm) or set to 0 for automatic optimization.
  target_ms2_tolerance: 10

  # Targeted optimization of ion mobility tolerance.
  # Use absolute values in 1/K0 (e.g. 0.04 1/K0) or set to 0 for automatic optimization.
  target_mobility_tolerance: 0

  # Targeted optimization of retention time tolerance.
  # Use absolute values in seconds (e.g. 300s) or set to 0 for automatic optimization.
  target_rt_tolerance: 0
```

## Calibration using LOESS
Individual properties like the retention time deviate from their library values and need to be calibrated (a). As a nonlinear but stable method, locally estimated scatterplot smoothing (LOESS) using both density and uniformly distributed kernels is used. (b) A collection of polynomial kernels is fitted to uniformly distributed subregions of the data. These consist of first and second degree polynomials basis functions of the calibratable property. (c) The individual functions are combined and smoothed using tricubic weights. (d) Combining the kernels with their weighting functions allows to approximate the systematic deviation of the data locally. (e), The sum of the weighted kernels can then be used for continuous approximation and calibration of retention times. The architecture is built on the scikit-learn package and can be configured to use different hyperparameters and arbitrary predictors for calibration.

<img src="../_static/images/methods_loess.png" width="100%" height="auto">

## Configuring the LOESS model

The type of model, the hyperparameters and the columns used as input and target for calibration can be set in the `calibration_manager` section of the configuration file.

```yaml
calibration_manager:
  - name: fragment
    estimators:
      - name: mz
        model: LOESSRegression
        model_args:
          n_kernels: 2
        input_columns:
          - mz_library
        target_columns:
          - mz_observed
        output_columns:
          - mz_calibrated
        # display deviation in ppm
        transform_deviation: 1e6
  - name: precursor
    estimators:
        - name: mz
          model: LOESSRegression
          model_args:
            n_kernels: 2
          input_columns:
            - mz_library
          target_columns:
            - mz_observed
          output_columns:
            - mz_calibrated
          # display deviation in ppm
          transform_deviation: 1e6
        - name: rt
          model: LOESSRegression
          model_args:
            n_kernels: 6
          input_columns:
            - rt_library
          target_columns:
            - rt_observed
          output_columns:
            - rt_calibrated
        - name: mobility
          model: LOESSRegression
          model_args:
            n_kernels: 2
          input_columns:
            - mobility_library
          target_columns:
            - mobility_observed
          output_columns:
            - mobility_calibrated

```
