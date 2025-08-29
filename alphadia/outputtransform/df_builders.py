import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase

from alphadia.constants.keys import StatOutputKeys
from alphadia.workflow.managers.calibration_manager import (
    CalibrationEstimators,
    CalibrationGroups,
    CalibrationManager,
)
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.managers.raw_file_manager import RawFileManager
from alphadia.workflow.managers.timing_manager import TimingManager
from alphadia.workflow.peptidecentric.peptidecentric import PeptideCentricWorkflow

logger = logging.getLogger()


def build_run_stat_df(
    folder: str,
    raw_name: str,
    run_df: pd.DataFrame,
    channels: list[int] | None = None,
):
    """Build stat dataframe for a single run.

    Parameters
    ----------

    folder: str
        Directory containing the raw file and the managers

    raw_name: str
        Name of the raw file

    run_df: pd.DataFrame
        Dataframe containing the precursor data

    channels: List[int], optional
        List of channels to include in the output, default=[0]

    Returns
    -------
    pd.DataFrame
        Dataframe containing the statistics

    """

    if channels is None:
        channels = [0]
    all_stats = []

    for channel in channels:
        channel_df = run_df[run_df["channel"] == channel]

        stats = {
            "run": raw_name,
            "channel": channel,
            "precursors": len(channel_df),
            "proteins": channel_df["pg"].nunique(),
        }

        stats["fwhm_rt"] = np.nan
        if "cycle_fwhm" in channel_df.columns:
            stats["fwhm_rt"] = np.mean(channel_df["cycle_fwhm"])

        stats["fwhm_mobility"] = np.nan
        if "mobility_fwhm" in channel_df.columns:
            stats["fwhm_mobility"] = np.mean(channel_df["mobility_fwhm"])

        # collect optimization stats
        optimization_stats = defaultdict(lambda: np.nan)
        if os.path.exists(
            optimization_manager_path := os.path.join(
                folder,
                PeptideCentricWorkflow.OPTIMIZATION_MANAGER_PKL_NAME,
            )
        ):
            optimization_manager = OptimizationManager(path=optimization_manager_path)
            optimization_stats[StatOutputKeys.MS2_ERROR] = (
                optimization_manager.ms2_error
            )
            optimization_stats[StatOutputKeys.MS1_ERROR] = (
                optimization_manager.ms1_error
            )
            optimization_stats[StatOutputKeys.RT_ERROR] = optimization_manager.rt_error
            optimization_stats[StatOutputKeys.MOBILITY_ERROR] = (
                optimization_manager.mobility_error
            )
        else:
            logger.warning(f"Error reading optimization manager for {raw_name}")

        for key in [
            StatOutputKeys.MS2_ERROR,
            StatOutputKeys.MS1_ERROR,
            StatOutputKeys.RT_ERROR,
            StatOutputKeys.MOBILITY_ERROR,
        ]:
            stats[f"{StatOutputKeys.OPTIMIZATION_PREFIX}{key}"] = optimization_stats[
                key
            ]

        # collect calibration stats
        calibration_stats = defaultdict(lambda: np.nan)
        if os.path.exists(
            calibration_manager_path := os.path.join(
                folder,
                PeptideCentricWorkflow.CALIBRATION_MANAGER_PKL_NAME,
            )
        ):
            calibration_manager = CalibrationManager(path=calibration_manager_path)

            if (
                fragment_mz_estimator := calibration_manager.get_estimator(
                    CalibrationGroups.FRAGMENT, CalibrationEstimators.MZ
                )
            ) and (fragment_mz_metrics := fragment_mz_estimator.metrics):
                calibration_stats["ms2_median_accuracy"] = fragment_mz_metrics[
                    "median_accuracy"
                ]
                calibration_stats["ms2_median_precision"] = fragment_mz_metrics[
                    "median_precision"
                ]

            if (
                precursor_mz_estimator := calibration_manager.get_estimator(
                    CalibrationGroups.PRECURSOR, CalibrationEstimators.MZ
                )
            ) and (precursor_mz_metrics := precursor_mz_estimator.metrics):
                calibration_stats["ms1_median_accuracy"] = precursor_mz_metrics[
                    "median_accuracy"
                ]
                calibration_stats["ms1_median_precision"] = precursor_mz_metrics[
                    "median_precision"
                ]

        else:
            logger.warning(f"Error reading calibration manager for {raw_name}")

        prefix = "calibration."
        for key in [
            "ms2_median_accuracy",
            "ms2_median_precision",
            "ms1_median_accuracy",
            "ms1_median_precision",
        ]:
            stats[f"{prefix}{key}"] = calibration_stats.get(key, "NaN")

        # collect raw stats
        raw_stats = defaultdict(lambda: np.nan)
        if os.path.exists(
            raw_file_manager_path := os.path.join(
                folder, PeptideCentricWorkflow.RAW_FILE_MANAGER_PKL_NAME
            )
        ):
            raw_stats = RawFileManager(
                path=raw_file_manager_path, load_from_file=True
            ).stats
        else:
            logger.warning(f"Error reading raw file manager for {raw_name}")

        # deliberately mapping explicitly to avoid coupling raw_stats to the output too tightly
        prefix = "raw."

        stats[f"{prefix}gradient_min_m"] = raw_stats["rt_limit_min"] / 60
        stats[f"{prefix}gradient_max_m"] = raw_stats["rt_limit_max"] / 60
        stats[f"{prefix}gradient_length_m"] = (
            raw_stats["rt_limit_max"] - raw_stats["rt_limit_min"]
        ) / 60
        for key in [
            "cycle_length",
            "cycle_duration",
            "cycle_number",
            "msms_range_min",
            "msms_range_max",
        ]:
            stats[f"{prefix}{key}"] = raw_stats[key]

        all_stats.append(stats)

    return pd.DataFrame(all_stats)


def build_run_internal_df(
    folder_path: str,
):
    """Build stat dataframe for a single run.

    Parameters
    ----------

    folder_path: str
        Path (from the base directory of the output_folder attribute of the SearchStep class) to the directory containing the raw file and the managers


    Returns
    -------
    pd.DataFrame
        Dataframe containing the statistics

    """
    timing_manager_path = os.path.join(
        folder_path, PeptideCentricWorkflow.TIMING_MANAGER_PKL_NAME
    )
    raw_name = os.path.basename(folder_path)

    internal_dict = {
        "run": [raw_name],
    }

    if os.path.exists(timing_manager_path):
        timing_manager = TimingManager(path=timing_manager_path)
        for key in timing_manager.timings:
            internal_dict[f"duration_{key}"] = [timing_manager.timings[key]["duration"]]

    else:
        logger.warning(f"Error reading timing manager for {raw_name}")

    return pd.DataFrame(internal_dict)


def transfer_library_stat_df(transfer_library: SpecLibBase) -> pd.DataFrame:
    """create statistics dataframe for transfer library

    Parameters
    ----------

    transfer_library : SpecLibBase
        transfer library

    Returns
    -------

    pd.DataFrame
        statistics dataframe
    """

    # get unique modifications
    modifications = (
        transfer_library.precursor_df["mods"].str.split(";").explode().unique()
    )
    modifications = [mod for mod in modifications if mod != ""]

    statistics_df = []
    for mod in modifications:
        mod_df = transfer_library.precursor_df[
            transfer_library.precursor_df["mods"].str.contains(mod)
        ]
        mod_ms2_df = mod_df[mod_df["use_for_ms2"]]
        statistics_df.append(
            {
                "modification": mod,
                "num_precursors": len(mod_df),
                "num_unique_precursor": len(mod_df["mod_seq_charge_hash"].unique()),
                "num_ms2_precursors": len(mod_ms2_df),
                "num_unique_ms2_precursor": len(
                    mod_ms2_df["mod_seq_charge_hash"].unique()
                ),
            }
        )

    # add unmodified
    mod_df = transfer_library.precursor_df[transfer_library.precursor_df["mods"] == ""]
    mod_ms2_df = mod_df[mod_df["use_for_ms2"]]
    statistics_df.append(
        {
            "modification": "",
            "num_precursors": len(mod_df),
            "num_unique_precursor": len(mod_df["mod_seq_charge_hash"].unique()),
            "num_ms2_precursors": len(mod_ms2_df),
            "num_unique_ms2_precursor": len(mod_ms2_df["mod_seq_charge_hash"].unique()),
        }
    )

    # add total
    statistics_df.append(
        {
            "modification": "Total",
            "num_precursors": len(transfer_library.precursor_df),
            "num_unique_precursor": len(
                transfer_library.precursor_df["mod_seq_charge_hash"].unique()
            ),
            "num_ms2_precursors": len(
                transfer_library.precursor_df[
                    transfer_library.precursor_df["use_for_ms2"]
                ]
            ),
            "num_unique_ms2_precursor": len(
                transfer_library.precursor_df[
                    transfer_library.precursor_df["use_for_ms2"]
                ]["mod_seq_charge_hash"].unique()
            ),
        }
    )

    return pd.DataFrame(statistics_df)


def log_stat_df(stat_df: pd.DataFrame):
    """log statistics dataframe to console

    Parameters
    ----------

    stat_df : pd.DataFrame
        statistics dataframe
    """

    # iterate over all modifications d
    # print with space padding
    space = 12
    logger.info(
        "Modification".ljust(25)
        + "Total".rjust(space)
        + "Unique".rjust(space)
        + "Total MS2".rjust(space)
        + "Unique MS2".rjust(space)
    )

    for _, row in stat_df.iterrows():
        if row["modification"] == "Total":
            continue
        logger.info(
            row["modification"].ljust(25)
            + f'{row["num_precursors"]:,}'.rjust(space)
            + f'{row["num_unique_precursor"]:,}'.rjust(space)
            + f'{row["num_ms2_precursors"]:,}'.rjust(space)
            + f'{row["num_unique_ms2_precursor"]:,}'.rjust(space)
        )
    # log line
    logger.info("-" * 25 + " " + "-" * space * 4)

    # log total
    total = stat_df[stat_df["modification"] == "Total"].iloc[0]
    logger.info(
        "Total".ljust(25)
        + f'{total["num_precursors"]:,}'.rjust(space)
        + f'{total["num_unique_precursor"]:,}'.rjust(space)
        + f'{total["num_ms2_precursors"]:,}'.rjust(space)
        + f'{total["num_unique_ms2_precursor"]:,}'.rjust(space)
    )
