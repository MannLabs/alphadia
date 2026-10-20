"""Sequence-dependent refinement of the calibrated retention time.

The RT calibration is a smooth function of the library RT, so a prediction error that depends on the peptide's
sequence (the library's RT model against this run's chromatography) is left in every precursor's calibrated RT. A
gradient-boosting model of the calibration identifications' residual on sequence descriptors removes most of it.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger()

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TERMINAL_POSITIONS = {"first": 0, "second": 1, "second_last": -2, "last": -1}
MAX_ITER = 300
LEARNING_RATE = 0.05
# fewer identifications than this do not determine the per-residue effects
MIN_IDENTIFICATIONS = 300
RANDOM_STATE = 42


def sequence_descriptors(precursor_df: pd.DataFrame) -> pd.DataFrame:
    """Amino-acid counts, length, terminal residues, modification counts, charge and calibrated RT."""
    sequence = precursor_df["sequence"]
    mods = precursor_df["mods"].fillna("")
    descriptors = pd.DataFrame(
        {f"n_{aa}": sequence.str.count(aa) for aa in AMINO_ACIDS},
        index=precursor_df.index,
    )
    descriptors["length"] = sequence.str.len()
    codes = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for name, position in TERMINAL_POSITIONS.items():
        descriptors[name] = sequence.str[position].map(codes)
    descriptors["oxidation"] = mods.str.count("Oxidation")
    descriptors["acetyl"] = mods.str.contains("Acetyl").astype(int)
    descriptors["charge"] = precursor_df["charge"]
    descriptors["rt_calibrated"] = precursor_df["rt_calibrated"]
    return descriptors


def refine_calibrated_rt(
    library_precursor_df: pd.DataFrame, identifications_df: pd.DataFrame
) -> None:
    """Add the predicted sequence-dependent RT error to `rt_calibrated` of every library precursor, in place.

    The model is fitted on the target identifications' residual `rt_observed - rt_calibrated`. Every precursor of an
    elution group gets the correction of the group's target, so a decoy keeps the RT of its target.

    Parameters
    ----------
    library_precursor_df : pd.DataFrame
        Library precursors with `precursor_idx`, `elution_group_idx`, `decoy`, `sequence`, `mods`, `charge` and the
        calibrated `rt_calibrated`.
    identifications_df : pd.DataFrame
        Target identifications with `precursor_idx` and `rt_observed`.
    """
    library = library_precursor_df.set_index("precursor_idx")
    ids = identifications_df[["precursor_idx", "rt_observed"]].drop_duplicates(
        "precursor_idx"
    )
    ids = ids.join(library, on="precursor_idx", how="inner")
    ids = ids[ids["decoy"] == 0]
    if len(ids) < MIN_IDENTIFICATIONS:
        logger.info(
            f"RT refinement skipped: {len(ids)} identifications, fewer than {MIN_IDENTIFICATIONS}"
        )
        return

    residual = (ids["rt_observed"] - ids["rt_calibrated"]).to_numpy()
    model = HistGradientBoostingRegressor(
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        categorical_features=list(TERMINAL_POSITIONS),
        random_state=RANDOM_STATE,
    )
    model.fit(sequence_descriptors(ids), residual)

    targets = library_precursor_df[library_precursor_df["decoy"] == 0]
    target_correction = pd.Series(
        model.predict(sequence_descriptors(targets)),
        index=targets["elution_group_idx"].to_numpy(),
    )
    target_correction = target_correction[~target_correction.index.duplicated()]
    correction = (
        library_precursor_df["elution_group_idx"].map(target_correction).fillna(0.0)
    )
    library_precursor_df["rt_calibrated"] += correction.to_numpy()

    fitted = residual - model.predict(sequence_descriptors(ids))
    logger.info(
        f"RT refinement on {len(ids):,} identifications: median |residual| "
        f"{np.median(np.abs(residual)):.2f} -> {np.median(np.abs(fitted)):.2f} s (in sample), "
        f"median |correction| {np.median(np.abs(correction)):.2f} s"
    )
