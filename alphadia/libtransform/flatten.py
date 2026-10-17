import logging

from alphabase.peptide import fragment
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia import utils
from alphadia.constants.keys import CalibCols
from alphadia.libtransform.base import ProcessingStep
from alphadia.validation.schemas import fragments_flat_schema, precursors_flat_schema

logger = logging.getLogger()

SHORT_ION_MAX_NUMBER = 3


class DropShortFragments(ProcessingStep):
    def __init__(self, max_number: int = SHORT_ION_MAX_NUMBER) -> None:
        """Zero the intensities of b and y ions up to `max_number` residues, so flattening leaves them out.

        Short ions of real peptides follow position-specific residue frequencies, so a false target's short ions
        coincide with those of co-eluting peptides more often than a decoy's do once the decoy changes the residues
        next to the termini. Not scoring them for anyone removes that chance-match advantage of false targets.
        """
        self.max_number = max_number
        super().__init__()

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        return isinstance(input, SpecLibBase)

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Zero the short-ion intensities of every precursor."""
        intensity_df = input._fragment_intensity_df
        start = input.precursor_df["frag_start_idx"].to_numpy()
        stop = input.precursor_df["frag_stop_idx"].to_numpy()
        b_columns = [i for i, c in enumerate(intensity_df.columns) if c.startswith("b")]
        y_columns = [i for i, c in enumerate(intensity_df.columns) if c.startswith("y")]
        for k in range(1, self.max_number + 1):
            # row i of a precursor holds b(i+1) and y(n-1-i); peptides too short for ion k have no such row
            has_ion = stop - start >= k
            b_rows = (start + k - 1)[has_ion]
            y_rows = (stop - k)[has_ion]
            intensity_df.iloc[b_rows, b_columns] = 0.0
            intensity_df.iloc[y_rows, y_columns] = 0.0
        logger.info(f"Zeroed b and y ions up to {self.max_number} residues")
        return input


class FlattenLibrary(ProcessingStep):
    def __init__(
        self, top_k_fragments: int = 12, min_fragment_intensity: float = 0.01
    ) -> None:
        """Convert a `SpecLibBase` object into a `SpecLibFlat` object.

        Parameters
        ----------
        top_k_fragments : int, optional
            Number of top fragments to keep. Default is 12.

        min_fragment_intensity : float, optional
            Minimum intensity threshold for fragments. Default is 0.01.

        """
        self.top_k_fragments = top_k_fragments
        self.min_fragment_intensity = min_fragment_intensity

        super().__init__()

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        return isinstance(input, SpecLibBase)

    def forward(self, input: SpecLibBase) -> SpecLibFlat:
        """Convert a `SpecLibBase` object into a `SpecLibFlat` object."""
        input._fragment_cardinality_df = fragment.calc_fragment_cardinality(  # type: ignore[attr-defined]
            input.precursor_df, input._fragment_mz_df
        )
        output = SpecLibFlat(
            min_fragment_intensity=self.min_fragment_intensity,
            keep_top_k_fragments=self.top_k_fragments,
        )
        output.parse_base_library(
            input,
            custom_df={"cardinality": input._fragment_cardinality_df},  # type: ignore[attr-defined]
        )

        return output


class InitFlatColumns(ProcessingStep):
    def __init__(self) -> None:
        """Initialize the columns of a `SpecLibFlat` object for alphadia search.
        Calibratable columns are `mz_library`, `rt_library` and `mobility_library` will be initialized with the first matching column in the input dataframe.
        """
        super().__init__()

    def validate(self, input: SpecLibFlat) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        return isinstance(input, SpecLibFlat)

    def forward(self, input: SpecLibFlat) -> SpecLibFlat:
        """Initialize the columns of a `SpecLibFlat` object for alphadia search."""
        precursor_columns = {
            CalibCols.MZ_LIBRARY: ["mz_library", "mz", "precursor_mz"],
            CalibCols.RT_LIBRARY: [
                "rt_library",
                "rt",
                "rt_norm",
                "rt_pred",
                "rt_norm_pred",
                "irt",
            ],
            CalibCols.MOBILITY_LIBRARY: [
                "mobility_library",
                "mobility",
                "mobility_pred",
            ],
        }

        fragment_columns = {
            CalibCols.MZ_LIBRARY: ["mz_library", "mz", "predicted_mz"],
        }

        for column_mapping, df in [
            (precursor_columns, input.precursor_df),
            (fragment_columns, input.fragment_df),
        ]:
            for key, value in column_mapping.items():
                for candidate_columns in value:
                    if candidate_columns in df.columns:
                        df.rename(columns={candidate_columns: key}, inplace=True)
                        # break after first match
                        break

        if CalibCols.MOBILITY_LIBRARY not in input.precursor_df.columns:
            input.precursor_df[CalibCols.MOBILITY_LIBRARY] = 0
            logger.warning("Library contains no ion mobility annotations")

        precursors_flat_schema.validate(input.precursor_df)
        fragments_flat_schema.validate(input.fragment_df)

        return input


class LogFlatLibraryStats(ProcessingStep):
    def __init__(self) -> None:
        """Log basic statistics of a `SpecLibFlat` object."""
        super().__init__()

    def validate(self, input: SpecLibFlat) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        return isinstance(input, SpecLibFlat)

    def forward(self, input: SpecLibFlat) -> SpecLibFlat:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        logger.info("============ Library Stats ============")
        logger.info(f"Number of precursors: {len(input.precursor_df):,}")

        if "decoy" in input.precursor_df.columns:
            n_targets = len(input.precursor_df.query("decoy == False"))
            n_decoys = len(input.precursor_df.query("decoy == True"))
            logger.info(f"\tthereof targets:{n_targets:,}")
            logger.info(f"\tthereof decoys: {n_decoys:,}")
        else:
            logger.warning("no decoy column was found")

        if "elution_group_idx" in input.precursor_df.columns:
            n_elution_groups = len(input.precursor_df["elution_group_idx"].unique())
            average_precursors_per_group = len(input.precursor_df) / n_elution_groups
            logger.info(f"Number of elution groups: {n_elution_groups:,}")
            logger.info(f"\taverage size: {average_precursors_per_group:.2f}")

        else:
            logger.warning("no elution_group_idx column was found")

        if "proteins" in input.precursor_df.columns:
            n_proteins = len(input.precursor_df["proteins"].unique())
            logger.info(f"Number of proteins: {n_proteins:,}")
        else:
            logger.warning("no proteins column was found")

        if "channel" in input.precursor_df.columns:
            channels = input.precursor_df["channel"].unique()
            n_channels = len(channels)
            logger.info(f"Number of channels: {n_channels:,} ({channels})")

        else:
            logger.warning("no channel column was found, will assume only one channel")

        isotopes = utils.get_isotope_columns(input.precursor_df.columns)

        if len(isotopes) > 0:
            logger.info(f"Isotopes Distribution for {len(isotopes)} isotopes")

        logger.info("=======================================")

        return input
