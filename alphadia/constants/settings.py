"""Internal constants."""

from alphadia.constants.keys import SearchTolerance

FIGURES_FOLDER_NAME = "figures"

NUM_FEATURES = 46
MAX_FRAGMENT_MZ_TOLERANCE = 200


class CalibrationLoop:
    """Constants of the calibration loop, see `OptimizationHandler.search_parameter_optimization`."""

    # A calibration fit needs at least this number of identifications.
    MIN_PRECURSORS_TO_RECALIBRATE = 6
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
    # The tolerances converged when no tolerance changes by more than this share between rounds.
    CONVERGENCE_THRESHOLD = 0.1
    # Restore the best checkpoint when a round on the same lock finds more than this share fewer identifications.
    MAX_IDENTIFICATION_LOSS = 0.05
