"""Internal constants."""

from alphadia.constants.keys import SearchTolerance

FIGURES_FOLDER_NAME = "figures"

NUM_FEATURES = 46
MAX_FRAGMENT_MZ_TOLERANCE = 200


class CalibrationLoop:
    """Constants of the calibration loop and its phases (see `OptimizationHandler.search_parameter_optimization`)."""

    # warm-up: recalibrate as soon as the lock holds this many precursors at 1% FDR, and grow the lock with the
    # calibrated tolerances until it reaches calibration.optimization_lock_target
    WARM_UP_MIN_PRECURSORS = 100
    # warm-up: a tolerance shrinks at most by this factor per round
    WARM_UP_MAX_SHRINK = 2
    # the tolerances cover this central interval of the calibrated residuals of the identifications at 1% FDR ...
    TOLERANCE_QUANTILE = 0.99
    # ... times this factor. Only automatic parameters use the factor. Targeted parameters take their target.
    # RT gets a wider margin. On Astral, factors from 3 to 6 gave about 0.8% more precursors than 1.5.
    TOLERANCE_FACTOR = {
        SearchTolerance.MS2_ERROR: 1.2,
        SearchTolerance.MS1_ERROR: 1.2,
        SearchTolerance.RT_ERROR: 3.0,
        SearchTolerance.MOBILITY_ERROR: 1.2,
    }
    # converge: converged when no tolerance changes by more than this share between rounds
    CONVERGENCE_THRESHOLD = 0.1
    # stop when a round identifies this share fewer precursors per elution group than the best calibrated round
    MAX_ID_LOSS = 0.05
