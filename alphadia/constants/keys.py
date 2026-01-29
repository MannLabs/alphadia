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


class StatOutputCols(metaclass=ConstantsClass):
    """String constants for reading and writing output columns for the `stat` file."""

    # optimization
    OPTIMIZATION_PREFIX = "optimization."
    MS1_ERROR = "ms1_error"
    MS2_ERROR = "ms2_error"
    RT_ERROR = "rt_error"
    MOBILITY_ERROR = "mobility_error"


class ConfigKeys(metaclass=ConstantsClass):
    """String constants for accessing the config."""

    VERSION = "version"
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


class InferenceStrategy(metaclass=ConstantsClass):
    """String constants for protein inference strategies."""

    LIBRARY = "library"
    MAXIMUM_PARSIMONY = "maximum_parsimony"
    HEURISTIC = "heuristic"


class QuantificationLevelName(metaclass=ConstantsClass):
    """String constants for accessing the quantification level."""

    PRECURSOR = "precursor"
    PEPTIDE = "peptide"
    PROTEIN = "pg"


class QuantificationLevelKey(metaclass=ConstantsClass):
    """String constants for accessing the quantification level key."""

    PRECURSOR = "mod_seq_charge_hash"
    PEPTIDE = "mod_seq_hash"
    PROTEIN = "pg"


class NormalizationMethods(metaclass=ConstantsClass):
    """String constants for LFQ methods."""

    DIRECTLFQ: str = "directlfq"
    QUANTSELECT: str = "quantselect"


class PrecursorOutputCols(metaclass=ConstantsClass):
    """String constants for accessing the precursor output columns."""

    # Core identification
    IDX = "precursor.idx"
    ELUTION_GROUP_IDX = "precursor.elution_group_idx"
    SEQUENCE = "precursor.sequence"
    CHARGE = "precursor.charge"
    MODS = "precursor.mods"
    MOD_SITES = "precursor.mod_sites"
    MOD_SEQ_HASH = "precursor.mod_seq_hash"
    MOD_SEQ_CHARGE_HASH = "precursor.mod_seq_charge_hash"

    RANK = "precursor.rank"
    NAA = "precursor.naa"

    # Mass measurements
    MZ_LIBRARY = "precursor.mz.library"
    MZ_OBSERVED = "precursor.mz.observed"
    MZ_CALIBRATED = "precursor.mz.calibrated"

    # Retention time measurements
    RT_LIBRARY = "precursor.rt.library"
    RT_OBSERVED = "precursor.rt.observed"
    RT_CALIBRATED = "precursor.rt.calibrated"
    RT_FWHM = "precursor.rt.fwhm"

    # Mobility measurements
    MOBILITY_LIBRARY = "precursor.mobility.library"
    MOBILITY_OBSERVED = "precursor.mobility.observed"
    MOBILITY_CALIBRATED = "precursor.mobility.calibrated"
    MOBILITY_FWHM = "precursor.mobility.fwhm"

    # Quantification
    INTENSITY = "precursor.intensity"

    # Quality scores
    QVAL = "precursor.qval"
    PROBA = "precursor.proba"
    SCORE = "precursor.score"

    # Experimental metadata
    CHANNEL = "precursor.channel"
    DECOY = "precursor.decoy"


class PeptideOutputCols(metaclass=ConstantsClass):
    """String constants for accessing the peptide output columns."""

    INTENSITY = "peptide.intensity"


class ProteinGroupOutputCols(metaclass=ConstantsClass):
    """String constants for accessing the protein group output columns."""

    NAME = "pg.name"
    PROTEINS = "pg.proteins"
    GENES = "pg.genes"
    MASTER_PROTEIN = "pg.master_protein"
    QVAL = "pg.qval"
    INTENSITY = "pg.intensity"


class OutputRawCols(metaclass=ConstantsClass):
    """String constants for experiment metadata columns.

    These keys represent experimental/run-level metadata that are not
    specific to precursors, peptides, or protein groups.
    """

    NAME = "raw.name"


class StatSearchCols(metaclass=ConstantsClass):
    """String constants for search statistics columns."""

    CHANNEL = "search.channel"
    PRECURSORS = "search.precursors"
    PROTEINS = "search.proteins"
    FWHM_RT = "search.fwhm_rt"
    FWHM_MOBILITY = "search.fwhm_mobility"


class StatCalibrationCols(metaclass=ConstantsClass):
    """String constants for calibration statistics columns."""

    MS2_BIAS = "calibration.ms2_bias"
    MS2_ERROR = "calibration.ms2_variance"
    MS1_BIAS = "calibration.ms1_bias"
    MS1_ERROR = "calibration.ms1_variance"


# this mapping is also used to filter the output columns, so only its values are kept
INTERNAL_TO_OUTPUT_MAPPING = {
    "peptide_lfq_intensity": PeptideOutputCols.INTENSITY,
    "precursor_lfq_intensity": PrecursorOutputCols.INTENSITY,
    "precursor_idx": PrecursorOutputCols.IDX,
    "elution_group_idx": PrecursorOutputCols.ELUTION_GROUP_IDX,
    "rank": PrecursorOutputCols.RANK,
    "naa": PrecursorOutputCols.NAA,
    "sequence": PrecursorOutputCols.SEQUENCE,
    "charge": PrecursorOutputCols.CHARGE,
    "mods": PrecursorOutputCols.MODS,
    "mod_sites": PrecursorOutputCols.MOD_SITES,
    "mod_seq_hash": PrecursorOutputCols.MOD_SEQ_HASH,
    "mod_seq_charge_hash": PrecursorOutputCols.MOD_SEQ_CHARGE_HASH,
    "mz_library": PrecursorOutputCols.MZ_LIBRARY,
    "mz_observed": PrecursorOutputCols.MZ_OBSERVED,
    "mz_calibrated": PrecursorOutputCols.MZ_CALIBRATED,
    "rt_library": PrecursorOutputCols.RT_LIBRARY,
    "rt_observed": PrecursorOutputCols.RT_OBSERVED,
    "rt_calibrated": PrecursorOutputCols.RT_CALIBRATED,
    "mobility_library": PrecursorOutputCols.MOBILITY_LIBRARY,
    "mobility_observed": PrecursorOutputCols.MOBILITY_OBSERVED,
    "mobility_calibrated": PrecursorOutputCols.MOBILITY_CALIBRATED,
    "qval": PrecursorOutputCols.QVAL,
    "proba": PrecursorOutputCols.PROBA,
    "score": PrecursorOutputCols.SCORE,
    "cycle_fwhm": PrecursorOutputCols.RT_FWHM,
    "mobility_fwhm": PrecursorOutputCols.MOBILITY_FWHM,
    "channel": PrecursorOutputCols.CHANNEL,
    "decoy": PrecursorOutputCols.DECOY,
    "pg": ProteinGroupOutputCols.NAME,
    "pg_lfq_intensity": ProteinGroupOutputCols.INTENSITY,
    "proteins": ProteinGroupOutputCols.PROTEINS,
    "genes": ProteinGroupOutputCols.GENES,
    "pg_master": ProteinGroupOutputCols.MASTER_PROTEIN,
    "pg_qval": ProteinGroupOutputCols.QVAL,
    "run": OutputRawCols.NAME,
}
