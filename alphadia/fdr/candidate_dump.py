"""Diagnostic files of the final extraction round, written next to the final-round feature matrix when
`fdr.save_feature_matrix` is set.

They show where a precursor was lost before the final classifier: every selected candidate with the raw-score cutoff
decision (`CANDIDATES_FILE_NAME`), every final-round row with the riding removal decision (`RIDING_FILE_NAME`), the
features of the rows the riding removal dropped (`RIDING_REMOVED_FEATURES_FILE_NAME`, same layout as the final-round
feature matrix, with the riding pre-score P(decoy) as `proba`), and the run's library precursors
(`LIBRARY_PRECURSORS_FILE_NAME`): precursor_idx and elution_group_idx are renumbered when the library is loaded, so
the other files cannot be mapped to the library file. Candidates and rows are keyed by (precursor_idx, rank).
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger()

CANDIDATES_FILE_NAME = "final_candidates.parquet"
RIDING_FILE_NAME = "final_riding.parquet"
RIDING_REMOVED_FEATURES_FILE_NAME = "riding_removed_features.parquet"
LIBRARY_PRECURSORS_FILE_NAME = "library_precursors.parquet"
COMPRESSION = "zstd"

# candidate columns of the selection and their stored types; missing ones (other backends) are left out
CANDIDATE_COLUMNS = {
    "precursor_idx": np.uint32,
    "rank": np.uint8,
    "elution_group_idx": np.uint32,
    "decoy": np.int8,
    "channel": np.int16,
    "score": np.float32,
    "selection_det_share": np.float32,
    "selection_n_detected": np.float32,
}
# the selection reports the apex as the first spectrum of its cycle, as the scoring does for rt_observed
APEX_FRAME_COLUMN = "frame_center"
RT_OBSERVED_COLUMN = "rt_observed"
ABOVE_CUTOFF_COLUMN = "above_cutoff"
GATE_ADMITTED_COLUMN = "det_share_admitted"
# constant columns of the candidate file: the selection and cutoff settings of the final extraction
SCORE_CUTOFF_COLUMN = "score_cutoff"
DET_SHARE_GATE_COLUMN = "det_share_gate"
RT_TOLERANCE_COLUMN = "rt_tolerance"
NUM_CANDIDATES_COLUMN = "num_candidates"

RIDING_ID_COLUMNS = {
    "precursor_idx": np.uint32,
    "rank": np.uint8,
    "decoy": np.int8,
}
PRESCORE_COLUMN = "prescore_proba"
# the riding removal result of every final-round row
RIDING_PARTNER_COLUMN = "riding_partner"
RIDING_LOSER_COLUMN = "riding_loser"
RIDING_SHARE_COLUMN = "riding_share"
RIDING_OWN_SUPPORT_COLUMN = "riding_own_support"
RIDING_REMOVED_COLUMN = "riding_removed"
RIDING_COLUMNS = [
    RIDING_PARTNER_COLUMN,
    RIDING_LOSER_COLUMN,
    RIDING_SHARE_COLUMN,
    RIDING_OWN_SUPPORT_COLUMN,
    RIDING_REMOVED_COLUMN,
]

LIBRARY_COLUMNS = [
    "precursor_idx",
    "elution_group_idx",
    "decoy",
    "channel",
    "sequence",
    "mods",
    "mod_sites",
    "charge",
    "proteins",
    "genes",
    "mz_library",
    "mz_calibrated",
    "rt_library",
    "rt_calibrated",
]


def write_candidates(
    folder: str,
    candidates_df: pd.DataFrame,
    above_cutoff: np.ndarray,
    kept: np.ndarray,
    rt_values: np.ndarray,
    settings: dict[str, float],
) -> None:
    """Write every candidate of `candidates_df` (the selection, before the raw-score cutoff) to `CANDIDATES_FILE_NAME`.

    `above_cutoff` marks the candidates whose raw score passed the cutoff and `kept` those that were scored (above the
    cutoff or admitted by the detection share); `rt_values` are the RTs of the spectra the apex frame indexes into. Each
    entry of `settings` (the cutoff, the gate, the selection tolerances) is stored as a constant column.
    """
    table = pd.DataFrame(
        {
            column: candidates_df[column].to_numpy().astype(dtype)
            for column, dtype in CANDIDATE_COLUMNS.items()
            if column in candidates_df.columns
        }
    )
    if APEX_FRAME_COLUMN in candidates_df.columns:
        frame = np.minimum(
            candidates_df[APEX_FRAME_COLUMN].to_numpy(np.int64), len(rt_values) - 1
        )
        table[RT_OBSERVED_COLUMN] = np.asarray(rt_values, dtype=np.float32)[frame]
    table[ABOVE_CUTOFF_COLUMN] = np.asarray(above_cutoff, dtype=bool)
    table[GATE_ADMITTED_COLUMN] = (
        np.asarray(kept, dtype=bool) & ~table[ABOVE_CUTOFF_COLUMN].to_numpy()
    )
    for name, value in settings.items():
        table[name] = np.float32(value)
    path = os.path.join(folder, CANDIDATES_FILE_NAME)
    table.to_parquet(path, index=False, compression=COMPRESSION)
    logger.info(f"Saved the {len(table):,} selected candidates to {path}")


def write_riding(
    folder: str,
    psm_df: pd.DataFrame,
    prescore: np.ndarray,
    riding_rows: pd.DataFrame | None,
) -> None:
    """Write every final-round row of `psm_df` with its pre-score P(decoy) `prescore` and the riding removal result
    `riding_rows` (row-aligned, from `fdr._riding_rows`; None if the removal found no partners) to `RIDING_FILE_NAME`.
    """
    table = pd.DataFrame(
        {
            column: psm_df[column].to_numpy().astype(dtype)
            for column, dtype in RIDING_ID_COLUMNS.items()
            if column in psm_df.columns
        }
    )
    table[PRESCORE_COLUMN] = np.asarray(prescore, dtype=np.float32)
    if riding_rows is None:
        riding_rows = empty_riding_rows(len(table))
    for column in RIDING_COLUMNS:
        table[column] = riding_rows[column].to_numpy()
    path = os.path.join(folder, RIDING_FILE_NAME)
    table.to_parquet(path, index=False, compression=COMPRESSION)
    logger.info(
        f"Saved the riding removal of {len(table):,} final-round rows to {path}"
    )


def empty_riding_rows(n_rows: int) -> pd.DataFrame:
    """The riding removal result of `n_rows` rows that no rule touched: no partner, share and support not computed."""
    return pd.DataFrame(
        {
            RIDING_PARTNER_COLUMN: np.zeros(n_rows, dtype=bool),
            RIDING_LOSER_COLUMN: np.zeros(n_rows, dtype=bool),
            RIDING_SHARE_COLUMN: np.full(n_rows, np.nan, dtype=np.float32),
            RIDING_OWN_SUPPORT_COLUMN: np.full(n_rows, np.nan, dtype=np.float32),
            RIDING_REMOVED_COLUMN: np.zeros(n_rows, dtype=bool),
        }
    )


def write_library_precursors(folder: str, precursor_df: pd.DataFrame) -> None:
    """Write the identity, sequence, charge, protein, m/z and RT columns of the run's library `precursor_df` to
    `LIBRARY_PRECURSORS_FILE_NAME`."""
    columns = [c for c in LIBRARY_COLUMNS if c in precursor_df.columns]
    path = os.path.join(folder, LIBRARY_PRECURSORS_FILE_NAME)
    precursor_df[columns].to_parquet(path, index=False, compression=COMPRESSION)
    logger.info(f"Saved the {len(precursor_df):,} library precursors to {path}")


def removed_features_path(folder: str) -> str:
    return os.path.join(folder, RIDING_REMOVED_FEATURES_FILE_NAME)
