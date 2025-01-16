# native imports
import logging

import numpy as np

# third party imports
import pandas as pd
import sklearn

import alphadia.fdrexperimental as fdrx

# alphadia imports
# alpha family imports
from alphadia import fragcomp
from alphadia.fdr_utils import get_q_values, keep_best, plot_fdr

logger = logging.getLogger()


def perform_fdr(
    classifier: sklearn.base.BaseEstimator,
    available_columns: list[str],
    df_target: pd.DataFrame,
    df_decoy: pd.DataFrame,
    competetive: bool = False,
    group_channels: bool = True,
    figure_path: str | None = None,
    neptune_run=None,
    df_fragments: pd.DataFrame | None = None,
    dia_cycle: np.ndarray = None,
    fdr_heuristic: float = 0.1,
    **kwargs,
):
    """Performs FDR calculation on a dataframe of PSMs

    Parameters
    ----------

    classifier : sklearn.base.BaseEstimator
        A classifier that implements the fit and predict_proba methods

    available_columns : list[str]
        A list of column names that are available for the classifier

    df_target : pd.DataFrame
        A dataframe of target PSMs

    df_decoy : pd.DataFrame
        A dataframe of decoy PSMs

    competetive : bool
        Whether to perform competetive FDR calculation where only the highest scoring PSM in a target-decoy pair is retained

    group_channels : bool
        Whether to group PSMs by channel before performing competetive FDR calculation

    figure_path : str, default=None
        The path to save the FDR plot to

    neptune_run : neptune.run.Run, default=None
        The neptune run to log the FDR plot to

    reuse_fragments : bool, default=True
        Whether to reuse fragments for different precursors

    dia_cycle : np.ndarray, default=None
        The DIA cycle as provided by alphatims

    fdr_heuristic : float, default=0.1
        The FDR heuristic to use for the initial selection of PSMs before fragment competition

    Returns
    -------

    psm_df : pd.DataFrame
        A dataframe of PSMs with q-values and probabilities.
        The columns `qval` and `proba` are added to the input dataframes.
    """
    target_len, decoy_len = len(df_target), len(df_decoy)
    df_target.dropna(subset=available_columns, inplace=True)
    df_decoy.dropna(subset=available_columns, inplace=True)
    target_dropped, decoy_dropped = (
        target_len - len(df_target),
        decoy_len - len(df_decoy),
    )

    if target_dropped > 0:
        logger.warning(f"dropped {target_dropped} target PSMs due to missing features")

    if decoy_dropped > 0:
        logger.warning(f"dropped {decoy_dropped} decoy PSMs due to missing features")

    if (
        np.abs(len(df_target) - len(df_decoy)) / ((len(df_target) + len(df_decoy)) / 2)
        > 0.1
    ):
        logger.warning(
            f"FDR calculation for {len(df_target)} target and {len(df_decoy)} decoy PSMs"
        )
        logger.warning(
            "FDR calculation may be inaccurate as there is more than 10% difference in the number of target and decoy PSMs"
        )

    X_target = df_target[available_columns].values
    X_decoy = df_decoy[available_columns].values
    y_target = np.zeros(len(X_target))
    y_decoy = np.ones(len(X_decoy))

    X = np.concatenate([X_target, X_decoy])
    y = np.concatenate([y_target, y_decoy])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2
    )

    classifier.fit(X_train, y_train)

    psm_df = pd.concat([df_target, df_decoy])

    psm_df["_decoy"] = y

    if competetive:
        group_columns = (
            ["elution_group_idx", "channel"]
            if group_channels
            else ["elution_group_idx"]
        )
    else:
        group_columns = ["precursor_idx"]

    psm_df["proba"] = classifier.predict_proba(X)[:, 1]
    psm_df.sort_values("proba", ascending=True, inplace=True)

    psm_df = get_q_values(psm_df, "proba", "_decoy")

    if dia_cycle is not None and dia_cycle.shape[2] <= 2:
        # use a FDR of 10% as starting point
        # if there are no PSMs with a FDR < 10% use all PSMs
        start_idx = psm_df["qval"].searchsorted(fdr_heuristic, side="left")
        if start_idx == 0:
            start_idx = len(psm_df)

        # make sure fragments are not reused
        if df_fragments is not None:
            if dia_cycle is None:
                raise ValueError(
                    "dia_cycle must be provided if reuse_fragments is False"
                )
            fragment_competition = fragcomp.FragmentCompetition()
            psm_df = fragment_competition(
                psm_df.iloc[:start_idx], df_fragments, dia_cycle
            )

    psm_df = keep_best(psm_df, group_columns=group_columns)
    psm_df = get_q_values(psm_df, "proba", "_decoy")

    plot_fdr(
        X_train,
        X_test,
        y_train,
        y_test,
        classifier,
        psm_df["qval"],
        figure_path=figure_path,
        neptune_run=neptune_run,
    )

    return psm_df


def perform_fdr_new(
    classifier: fdrx.Classifier,
    available_columns: list[str],
    df: pd.DataFrame,
    group_columns,
    **kwargs,
):
    df.dropna(subset=available_columns, inplace=True)
    psm_df = classifier.fit_predict(
        df,
        available_columns + ["score"],
        "decoy",
        group_columns,
    )
    return psm_df
