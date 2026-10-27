import logging

import numpy as np
from alphabase.peptide import fragment
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia import utils
from alphadia.constants.keys import CalibCols
from alphadia.libtransform.base import ProcessingStep
from alphadia.validation.schemas import fragments_flat_schema, precursors_flat_schema

logger = logging.getLogger()


class FragmentLengthFilter(ProcessingStep):
    def __init__(self, min_fragment_residues: int = 0) -> None:
        """Remove fragments comprising fewer than `min_fragment_residues` residues from a `SpecLibBase` object.

        Their library intensity is set to zero, so the next most intense fragments take their place when the top k
        fragments are chosen.

        Parameters
        ----------
        min_fragment_residues : int, optional
            Minimum number of residues of a fragment. Default is 0, which keeps all fragments.

        """
        self.min_fragment_residues = min_fragment_residues

        super().__init__()

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        return isinstance(input, SpecLibBase)

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Set the intensity of fragments with fewer than `min_fragment_residues` residues to zero."""
        # every fragment comprises at least one residue
        if self.min_fragment_residues <= 1:
            return input

        frag_start = input.precursor_df["frag_start_idx"].to_numpy()
        n_frag_rows = input.precursor_df["frag_stop_idx"].to_numpy() - frag_start
        # row i of a precursor holds the N-terminal fragments with i + 1 residues and the C-terminal ones with
        # n_frag_rows - i residues
        row_offset = np.arange(n_frag_rows.sum()) - np.repeat(
            np.cumsum(n_frag_rows) - n_frag_rows, n_frag_rows
        )
        row_index = np.repeat(frag_start, n_frag_rows) + row_offset
        residues_by_direction = {
            1: row_offset + 1,
            -1: np.repeat(n_frag_rows, n_frag_rows) - row_offset,
        }

        intensity_df = input._fragment_intensity_df
        n_removed = 0
        for charged_frag_type in intensity_df.columns:
            frag_type, _ = fragment.parse_charged_frag_type(charged_frag_type)
            direction = fragment.FRAGMENT_TYPES[frag_type].direction_id
            if direction not in residues_by_direction:
                continue
            short_rows = row_index[
                residues_by_direction[direction] < self.min_fragment_residues
            ]
            intensity = intensity_df[charged_frag_type].to_numpy(copy=True)
            n_removed += int((intensity[short_rows] > 0).sum())
            intensity[short_rows] = 0
            intensity_df[charged_frag_type] = intensity

        logger.info(
            f"Removed {n_removed:,} fragments with fewer than {self.min_fragment_residues} residues"
        )
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
