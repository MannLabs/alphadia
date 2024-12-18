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


class OutputKeys(metaclass=ConstantsClass):
    """String constants for reading and writing output columns."""

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
