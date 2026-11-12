"""Internal constants."""

from alphadia.constants.keys import SearchTolerance

FIGURES_FOLDER_NAME = "figures"

NUM_FEATURES = 46
MAX_FRAGMENT_MZ_TOLERANCE = 200


class CalibrationLoop:
    """Constants of the calibration loop and its phases, see `OptimizationHandler.search_parameter_optimization`."""

    # Warm-up: calibrate as soon as the lock holds this number of identifications. Then grow the lock with the
    # calibrated tolerances until it reaches calibration.optimization_lock_target.
    WARM_UP_MIN_IDENTIFICATIONS = 100
    # Warm-up: a tolerance shrinks by at most this factor per round.
    WARM_UP_MAX_SHRINK = 2
    # The tolerances cover this central interval of the calibrated residuals of the identifications ...
    TOLERANCE_QUANTILE = 0.99
    # ... times this factor. Only automatic parameters use the factor. Targeted parameters take their target.
    # RT gets a wider margin. On Astral, factors from 3 to 6 gave about 0.8% more precursors than 1.5.
    TOLERANCE_FACTOR = {
        SearchTolerance.MS2_ERROR: 1.2,
        SearchTolerance.MS1_ERROR: 1.2,
        SearchTolerance.RT_ERROR: 3.0,
        SearchTolerance.MOBILITY_ERROR: 1.2,
    }
    # Converge: the tolerances converged when no tolerance changes by more than this share between rounds.
    CONVERGENCE_THRESHOLD = 0.1
    # Stop when the identification rate of a round is more than this share below the best checkpoint.
    MAX_IDENTIFICATION_LOSS = 0.05
