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


class InferenceStrategy(metaclass=ConstantsClass):
    """String constants for protein inference strategies."""

    LIBRARY = "library"
    MAXIMUM_PARSIMONY = "maximum_parsimony"
    HEURISTIC = "heuristic"


class SemanticPrecursorKeys(metaclass=ConstantsClass):
    """String constants for precursor output columns.

    These keys define the user-facing API for precursor data in output files.
    All precursor-specific properties use the 'precursor.' prefix for clarity.
    """

    # Core identification
    PRECURSOR_IDX = "precursor.idx"
    SEQUENCE = "precursor.sequence"
    CHARGE = "precursor.charge"
    MODS = "precursor.mods"
    MOD_SITES = "precursor.mod_sites"
    MOD_SEQ_HASH = "precursor.mod_seq_hash"
    MOD_SEQ_CHARGE_HASH = "precursor.mod_seq_charge_hash"

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


class SemanticPeptideKeys(metaclass=ConstantsClass):
    """String constants for peptide output columns.

    These keys define the user-facing API for peptide-level aggregation.
    """

    INTENSITY = "peptide.intensity"


class SemanticProteinGroupKeys(metaclass=ConstantsClass):
    """String constants for protein group output columns.

    These keys define the user-facing API for protein group data.
    The 'pg' identifier itself has no prefix for convenience as a grouping key.
    """

    PG = "pg"
    PROTEINS = "pg.proteins"
    GENES = "pg.genes"
    MASTER_PROTEIN = "pg.master_protein"
    QVAL = "pg.qval"
    INTENSITY = "pg.intensity"


class SemanticRawKeys(metaclass=ConstantsClass):
    """String constants for experimental metadata columns.

    These keys represent experimental/run-level metadata that are not
    specific to precursors, peptides, or protein groups.
    """

    RAW_NAME = "raw.name"


INTERNAL_TO_SEMANTIC_MAPPING = {
    "precursor_idx": SemanticPrecursorKeys.PRECURSOR_IDX,
    "sequence": SemanticPrecursorKeys.SEQUENCE,
    "charge": SemanticPrecursorKeys.CHARGE,
    "mods": SemanticPrecursorKeys.MODS,
    "mod_sites": SemanticPrecursorKeys.MOD_SITES,
    "mod_seq_hash": SemanticPrecursorKeys.MOD_SEQ_HASH,
    "mod_seq_charge_hash": SemanticPrecursorKeys.MOD_SEQ_CHARGE_HASH,
    "mz_library": SemanticPrecursorKeys.MZ_LIBRARY,
    "mz_observed": SemanticPrecursorKeys.MZ_OBSERVED,
    "mz_calibrated": SemanticPrecursorKeys.MZ_CALIBRATED,
    "rt_library": SemanticPrecursorKeys.RT_LIBRARY,
    "rt_observed": SemanticPrecursorKeys.RT_OBSERVED,
    "rt_calibrated": SemanticPrecursorKeys.RT_CALIBRATED,
    "mobility_library": SemanticPrecursorKeys.MOBILITY_LIBRARY,
    "mobility_observed": SemanticPrecursorKeys.MOBILITY_OBSERVED,
    "mobility_calibrated": SemanticPrecursorKeys.MOBILITY_CALIBRATED,
    "qval": SemanticPrecursorKeys.QVAL,
    "proba": SemanticPrecursorKeys.PROBA,
    "score": SemanticPrecursorKeys.SCORE,
    "cycle_fwhm": SemanticPrecursorKeys.RT_FWHM,
    "mobility_fwhm": SemanticPrecursorKeys.MOBILITY_FWHM,
    "channel": SemanticPrecursorKeys.CHANNEL,
    "decoy": SemanticPrecursorKeys.DECOY,
    "pg": SemanticProteinGroupKeys.PG,
    "proteins": SemanticProteinGroupKeys.PROTEINS,
    "genes": SemanticProteinGroupKeys.GENES,
    "pg_master": SemanticProteinGroupKeys.MASTER_PROTEIN,
    "pg_qval": SemanticProteinGroupKeys.QVAL,
    "run": SemanticRawKeys.RAW_NAME,
}
