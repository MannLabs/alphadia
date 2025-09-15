"""Module containing custom exceptions."""


class CustomError(Exception):
    """Custom alphaDIA error class."""

    _error_code = ""
    _msg = ""
    _detail_msg = ""
    _user_msg = ""

    @property
    def error_code(self):
        return self._error_code

    @property
    def msg(self):
        return self._msg

    @property
    def detail_msg(self):
        return self._detail_msg

    def __init__(self, msg: str = ""):
        self._user_msg = msg

        super().__init__(self._msg)

    def __str__(self):
        return (
            f"{self._error_code}: {self._msg}\n'{self._user_msg}'\n{self._detail_msg}"
        )


class BusinessError(CustomError):
    """Custom error class for 'business' errors.

    A 'business' error is an error that is caused during processing the input (data, configuration, ...) and not by a
    malfunction in AlphaDIA.
    """


class UserError(CustomError):
    """Custom error class for 'user' errors.

    A 'user' error is an error that is caused by the incompatible user input (data, configuration, ...) and not by a
    malfunction in AlphaDIA.
    """


class GenericUserError(UserError):
    """Raise when something is wrong with the user."""

    _error_code = "USER_ERROR"

    def __init__(self, msg: str, detail_msg: str = ""):
        self._msg = msg
        self._detail_msg = detail_msg


class NoPsmFoundError(BusinessError):
    """Raise when no PSMs are found in the search results."""

    _error_code = "NO_PSM_FOUND"

    _msg = "No psm files accumulated, can't continue."


class TooFewPSMError(BusinessError):
    """Raise when too few PSMs are available to perform a task."""

    _error_code = "TOO_FEW_PSM"

    _msg = "Too few PSMs available to perform the task."


class TooFewProteinsError(BusinessError):
    """Raise when too few Proteins are available to perform a task."""

    _error_code = "TOO_FEW_PROTEINS"

    _msg = "Too few proteins available to perform the task."


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


class NotValidDiaDataError(BusinessError):
    """Raise when data is not from DIA or invalid DIA data."""

    _error_code = "NOT_VALID_DIA_DATA"

    _msg = "Not valid DIA data."

    _detail_msg = """Please check if this is a valid DIA data set.

    1. This could be DDA data?
    2. Is it DIA data where the order of windows is not the same in each cycle?
    3. Might be due to DIA data where the cycle does not start at beginning of file but e.g. only after a series of blank scans.

    AlphaDIA has a strict definition of DIA cycles: same order of scans within a cycle and the cycle must appear throughout the whole file without gaps.
    """


class NoLibraryAvailableError(UserError):
    """Raise when no library is available."""

    _error_code = "NO_LIBRARY_AVAILABLE"

    _msg = "No spectral library available."

    _detail_msg = """No spectral library available.

    Either provide a spectral library file (via GUI, config or command line parameter),
    or provide a FASTA file and set the "Predict Library" (GUI) / 'library_prediction.predict' (config) flag."""


class ConfigError(BusinessError):
    """Raise when something is wrong with the provided configuration."""

    _error_code = "CONFIG_ERROR"

    _msg = "Malformed or invalid configuration."
    _key = ""
    _config_name = ""
    _detail_msg = ""

    def __init__(
        self,
        key: str = "",
        value: str = "",
        config_name: str = "",
        detail_msg: str = "",
    ):
        self._key = key
        self._value = value
        self._config_name = config_name
        self._detail_msg = detail_msg


class KeyAddedConfigError(ConfigError):
    """Raise when a key should be added to a config."""

    def __init__(self, key: str, value: str, config_name: str):
        super().__init__(key, value, config_name)
        self._detail_msg = (
            f"Defining new keys is not allowed when updating a config: "
            f"key='{self._key}', value='{self._value}', config_name='{self._config_name}'"
        )


class TypeMismatchConfigError(ConfigError):
    """Raise when the type of a value does not match the default type."""

    def __init__(self, key: str, value: str, config_name: str, extra_msg: str):
        super().__init__(key, value, config_name)
        self._detail_msg = (
            f"Types of values must match default config: "
            f"key='{self._key}', value='{self._value}', config_name='{self._config_name}', types='{extra_msg}'"
        )
