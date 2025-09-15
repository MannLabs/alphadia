class ConstantsClass(type):
    """A metaclass for classes that should only contain string constants."""

    def __setattr__(self, name, value):
        raise TypeError("Constants class cannot be modified")

    def get_values(cls):
        """Get all user-defined string values of the class."""
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]


class StatOutputKeys(metaclass=ConstantsClass):
    """String constants for reading and writing output columns for the `stat` file."""

    # optimization
    OPTIMIZATION_PREFIX = "optimization."
    MS1_ERROR = "ms1_error"
    MS2_ERROR = "ms2_error"
    RT_ERROR = "rt_error"
    MOBILITY_ERROR = "mobility_error"


class ConfigKeys(metaclass=ConstantsClass):
    """String constants for accessing the config."""

    OUTPUT_DIRECTORY = "output_directory"
    LIBRARY_PATH = "library_path"
    RAW_PATHS = "raw_paths"
    FASTA_PATHS = "fasta_paths"
    QUANT_DIRECTORY = "quant_directory"

    GENERAL = "general"
    SAVE_FIGURES = "save_figures"


class CalibCols(metaclass=ConstantsClass):
    """String constants for accessing (mz, rt, mobility) columns in the context of calibration."""

    MZ_OBSERVED = "mz_observed"
    MZ_LIBRARY = "mz_library"
    MZ_CALIBRATED = "mz_calibrated"

    RT_OBSERVED = "rt_observed"
    RT_LIBRARY = "rt_library"
    RT_CALIBRATED = "rt_calibrated"

    MOBILITY_OBSERVED = "mobility_observed"
    MOBILITY_LIBRARY = "mobility_library"
    MOBILITY_CALIBRATED = "mobility_calibrated"


class SearchStepFiles(metaclass=ConstantsClass):
    PSM_FILE_NAME = "psm.parquet"
    FRAG_FILE_NAME = "frag.parquet"
    FRAG_TRANSFER_FILE_NAME = "frag.transfer.parquet"
