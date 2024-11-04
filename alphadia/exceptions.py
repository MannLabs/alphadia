"""Module containing custom exceptions."""


class CustomError(Exception):
    """Custom alphaDIA error class."""

    _error_code = None
    _msg = None
    _detail_msg = ""

    @property
    def error_code(self):
        return self._error_code

    @property
    def msg(self):
        return self._msg

    @property
    def detail_msg(self):
        return self._detail_msg


class BusinessError(CustomError):
    """Custom error class for 'business' errors.

    A 'business' error is an error that is caused by the input (data, configuration, ...) and not by a
    malfunction in alphaDIA.
    """


class NoPsmFoundError(BusinessError):
    """Raise when no PSMs are found in the search results."""

    _error_code = "NO_PSM_FOUND"

    _msg = "No psm files accumulated, can't continue"


class NoOptimizationLockTargetError(BusinessError):
    """Raise when the optimization lock target is not found."""

    _error_code = "NO_OPTIMIZATION_LOCK_TARGET"

    _msg = "Searched all data without finding optimization lock target"

    _detail_msg = """Search for raw file failed as not enough precursors were found for calibration and optimization.
                 This can have the following reasons:
                   1. The sample was empty and therefore no precursors were found.
                   2. The sample contains only very few precursors.
                      For small libraries, try to set recalibration_target to a lower value.
                      For large libraries, try to reduce the library size and reduce the initial MS1 and MS2 tolerance.
                   3. There was a fundamental issue with search parameters."""


class NotDiaDataError(BusinessError):
    """Raise when data is not from DIA."""

    _error_code = "NOT_DIA_DATA"

    _msg = "Could not find cycle shape. Please check if this is a valid DIA data set."
