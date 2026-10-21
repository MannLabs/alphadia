"""Sequence-dependent refinement of the calibrated retention time.

The RT calibration is a smooth function of the library RT, so a prediction error that depends on the peptide's
sequence (the library's RT model against this run's chromatography) is left in every precursor's calibrated RT. A
gradient-boosting model of the calibration identifications' residual on sequence descriptors removes most of it.

The model is cross-fitted: an identification gets the prediction of the fold that did not see it, because an in-sample
fit on ~1k plasma identifications cut their residual from 10 to 2 s while held-out identifications only reached 7 s.
Where the cross-validated residual is not clearly smaller than the uncorrected one the refinement is skipped; on some
low-input files the correction made held-out identifications worse.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

logger = logging.getLogger()

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TERMINAL_POSITIONS = {"first": 0, "second": 1, "second_last": -2, "last": -1}
MAX_ITER = 300
LEARNING_RATE = 0.05
# fewer identifications than this do not determine the per-residue effects
MIN_IDENTIFICATIONS = 300
RANDOM_STATE = 42
N_FOLDS = 5
MAX_CROSS_VALIDATED_RESIDUAL_RATIO = 0.9


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
    id_descriptors = sequence_descriptors(ids)
    targets = library_precursor_df[library_precursor_df["decoy"] == 0]
    target_descriptors = sequence_descriptors(targets)

    out_of_fold = np.zeros_like(residual)
    target_prediction = np.zeros(len(targets))
    # folds split by sequence, so the other charge states of a peptide do not leak its RT into its own prediction
    folds = GroupKFold(N_FOLDS)
    for train, test in folds.split(id_descriptors, groups=ids["sequence"]):
        model = HistGradientBoostingRegressor(
            max_iter=MAX_ITER,
            learning_rate=LEARNING_RATE,
            categorical_features=list(TERMINAL_POSITIONS),
            random_state=RANDOM_STATE,
        )
        model.fit(id_descriptors.iloc[train], residual[train])
        out_of_fold[test] = model.predict(id_descriptors.iloc[test])
        target_prediction += model.predict(target_descriptors) / N_FOLDS

    uncorrected = np.median(np.abs(residual))
    cross_validated = np.median(np.abs(residual - out_of_fold))
    if cross_validated > MAX_CROSS_VALIDATED_RESIDUAL_RATIO * uncorrected:
        logger.info(
            f"RT refinement skipped: cross-validated median |residual| {cross_validated:.2f} s vs "
            f"{uncorrected:.2f} s uncorrected on {len(ids):,} identifications"
        )
        return

    target_correction = pd.Series(
        target_prediction, index=targets["precursor_idx"].to_numpy()
    )
    target_correction.loc[ids["precursor_idx"].to_numpy()] = out_of_fold
    is_identification = target_correction.index.isin(ids["precursor_idx"])
    target_correction.index = targets["elution_group_idx"].to_numpy()
    # an elution group with an identification takes that identification's out-of-fold correction
    target_correction = target_correction.iloc[
        np.argsort(~is_identification, kind="stable")
    ]
    target_correction = target_correction[~target_correction.index.duplicated()]
    correction = (
        library_precursor_df["elution_group_idx"].map(target_correction).fillna(0.0)
    )
    library_precursor_df["rt_calibrated"] += correction.to_numpy()

    logger.info(
        f"RT refinement on {len(ids):,} identifications: median |residual| "
        f"{uncorrected:.2f} -> {cross_validated:.2f} s (cross-validated), "
        f"median |correction| {np.median(np.abs(correction)):.2f} s"
    )
