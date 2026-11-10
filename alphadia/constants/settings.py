"""Internal constants."""

FIGURES_FOLDER_NAME = "figures"

NUM_FEATURES = 46
MAX_FRAGMENT_MZ_TOLERANCE = 200


class ConvergeCalibration:
    """Constants of the `converge` calibration strategy (see `OptimizationHandler._converge_search_parameters`)."""

    # fewer identifications than this give no usable calibration fit
    MIN_PRECURSORS_TO_RECALIBRATE = 6
    # the tolerances cover this central interval of the calibrated residuals of the identifications at 1% FDR ...
    TOLERANCE_QUANTILE = 0.99
    # ... times this factor (automatic parameters only; targeted parameters take their target).
    # RT gets a wider margin: on Astral, factors 3-6 gave about 0.8 % more precursors than 1.5.
    TOLERANCE_FACTOR = {
        "ms2_error": 1.2,
        "ms1_error": 1.2,
        "rt_error": 3.0,
        "mobility_error": 1.2,
    }
    # converged when no tolerance changes by more than this share between rounds
    CONVERGENCE_THRESHOLD = 0.1
    # restore the best earlier round when a round on the same lock finds more than this share fewer identifications
    MAX_ID_LOSS = 0.05
