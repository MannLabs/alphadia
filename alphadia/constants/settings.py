"""Internal constants."""

from alphadia.constants.keys import SearchTolerance

FIGURES_FOLDER_NAME = "figures"

NUM_FEATURES = 46
MAX_FRAGMENT_MZ_TOLERANCE = 200


class CalibrationLoop:
    """Constants of the calibration loop and its phases, see `OptimizationHandler.search_parameter_optimization`."""

    # An identification is a target precursor below this q-value.
    IDENTIFICATION_FDR = 0.01
    # Warm-up: calibrate as soon as the lock holds this number of identifications. Then grow the lock with the
    # calibrated tolerances until it reaches calibration.optimization_lock_target.
    WARM_UP_MIN_IDENTIFICATIONS = 100
    # Warm-up: a tolerance shrinks by at most this factor per round.
    WARM_UP_MAX_SHRINK = 2
    # The tolerances cover this central interval of the calibrated residuals of the identifications ...
    TOLERANCE_QUANTILE = 0.99
    # ... times this factor. Only automatic parameters use the factor. Targeted parameters take their target.
    # The calibration rounds search RT 3 times wider than the identifications, so that the last search shows the tail of
    # the true matches. The main search takes the RT interval of the true matches, see `true_match_interval`.
    TOLERANCE_FACTOR = {
        SearchTolerance.MS2_ERROR: 1.2,
        SearchTolerance.MS1_ERROR: 1.2,
        SearchTolerance.RT_ERROR: 3.0,
        SearchTolerance.MOBILITY_ERROR: 1.2,
    }
    # The RT tolerance of the main search spans at least this number of DIA cycles, so that it covers the sampling of
    # the elution peak even when the RT prediction is near perfect, as in an MBR step.
    MIN_RT_TOLERANCE_CYCLES = 10
    # The true match interval needs at least this expected number of true matches, the sum of 1 - PEP.
    MIN_EXPECTED_TRUE_MATCHES = 3
    # The classifier is refit on the last search at most this number of times, see `_converge_classifier`.
    MAX_CLASSIFIER_REFITS = 10
    # Converge: the tolerances converged when no tolerance changes by more than this share between rounds.
    CONVERGENCE_THRESHOLD = 0.1
    # Stop when the identification rate of a round is more than this share below the best checkpoint.
    MAX_IDENTIFICATION_LOSS = 0.05


class PeakGroupScoreCutoff:
    """The minimum peak group score of the main search is a factor times a low percentile of the scores of the
    calibration identifications, see `peak_parameters`."""

    FACTOR = 0.99
    PERCENTILE = 1
    # With `search.optimized_peak_group_score`, a higher cutoff saves memory and runtime at a small loss of precursors.
    MEMORY_SAVING_FACTOR = 0.95
    MEMORY_SAVING_PERCENTILE = 3
