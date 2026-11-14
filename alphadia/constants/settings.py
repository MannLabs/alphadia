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
    # The calibration rounds search RT 3 times wider than the identifications, so that the last round shows the tail of
    # the true matches. The main search takes the RT interval of the true matches, see `true_match_interval`.
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
