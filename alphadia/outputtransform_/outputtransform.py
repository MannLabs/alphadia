# native imports
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from alphabase.peptide import fragment, precursor
from alphabase.spectral_library import base
from alphabase.spectral_library.base import SpecLibBase
from outputtransform_.quant_builder import QuantBuilder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from alphadia import fdr, grouping, libtransform, utils
from alphadia.consensus.utils import read_df, write_df
from alphadia.constants.keys import ConfigKeys, StatOutputKeys
from alphadia.constants.settings import FIGURES_FOLDER_NAME
from alphadia.exceptions import NoPsmFoundError, TooFewProteinsError
from alphadia.fdrx.utils import train_test_split_
from alphadia.outputaccumulator import (
    AccumulationBroadcaster,
    TransferLearningAccumulator,
)
from alphadia.transferlearning.train import FinetuneManager
from alphadia.workflow import manager, peptidecentric
from alphadia.workflow.config import Config
from alphadia.workflow.managers.raw_file_manager import RawFileManager

logger = logging.getLogger()


class SearchPlanOutput:
    PSM_INPUT = "psm"
    PRECURSOR_OUTPUT = "precursors"
    STAT_OUTPUT = "stat"
    INTERNAL_OUTPUT = "internal"
    PG_OUTPUT = "protein_groups"
    LIBRARY_OUTPUT = "speclib.mbr"
    TRANSFER_OUTPUT = "speclib.transfer"
    TRANSFER_MODEL = "peptdeep.transfer"
    TRANSFER_STATS_OUTPUT = "stats.transfer"

    def __init__(self, config: Config, output_folder: str):
        """Combine individual searches into and build combined outputs

        In alphaDIA the search plan orchestrates the library building preparation,
        schedules the individual searches and combines the individual outputs into a single output.

        The SearchPlanOutput class is responsible for combining the individual search outputs into a single output.

        This includes:
        - combining the individual precursor tables
        - building the output stat table
        - performing protein grouping
        - performing protein FDR
        - performin label-free quantification
        - building the spectral library

        Parameters
        ----------

        config: dict
            Configuration dictionary

        output_folder: str
            Output folder
        """
        self.config = config
        self.output_folder = output_folder

        self._figure_path = (
            os.path.join(self.output_folder, FIGURES_FOLDER_NAME)
            if self.config[ConfigKeys.GENERAL][ConfigKeys.SAVE_FIGURES]
            else None
        )
        if self._figure_path and not os.path.exists(self._figure_path):
            os.makedirs(self._figure_path)

    def build(
        self,
        folder_list: list[str],
        base_spec_lib: base.SpecLibBase,
    ):
        """Build output from a list of seach outputs
        The following files are written to the output folder:
        - precursor.tsv
        - protein_groups.tsv
        - stat.tsv
        - speclib.mbr.hdf

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        base_spec_lib: base.SpecLibBase
            Base spectral library

        """
        logger.progress("Processing search outputs")
        psm_df = self.build_precursor_table(
            folder_list, save=False, base_spec_lib=base_spec_lib
        )
        _ = self.build_stat_df(folder_list, psm_df=psm_df, save=True)
        _ = self.build_internal_df(folder_list, save=True)
        _ = self.build_lfq_tables(folder_list, psm_df=psm_df, save=True)
        _ = self.build_library(
            base_spec_lib,
            psm_df=psm_df,
        )

        if self.config["transfer_library"]["enabled"]:
            _ = self.build_transfer_library(folder_list, save=True)

        if self.config["transfer_learning"]["enabled"]:
            _ = self.build_transfer_model(save=True)

    def build_transfer_model(self, save=True):
        """
        Finetune PeptDeep models using the transfer library

        Parameters
        ----------
        save : bool, optional
            Whether to save the statistics of the transfer learning on disk, by default True
        """
        logger.progress("Train PeptDeep Models")

        transfer_lib_path = os.path.join(
            self.output_folder, f"{self.TRANSFER_OUTPUT}.hdf"
        )
        if not os.path.exists(transfer_lib_path):
            raise ValueError(
                f"Transfer library not found at {transfer_lib_path}, did you enable library generation?"
            )

        transfer_lib = SpecLibBase()
        transfer_lib.load_hdf(
            transfer_lib_path,
            load_mod_seq=True,
        )

        device = utils.get_torch_device(self.config["general"]["use_gpu"])

        tune_mgr = FinetuneManager(
            device=device,
            lr_patience=self.config["transfer_learning"]["lr_patience"],
            test_interval=self.config["transfer_learning"]["test_interval"],
            train_fraction=self.config["transfer_learning"]["train_fraction"],
            validation_fraction=self.config["transfer_learning"]["validation_fraction"],
            test_fraction=self.config["transfer_learning"]["test_fraction"],
            epochs=self.config["transfer_learning"]["epochs"],
            warmup_epochs=self.config["transfer_learning"]["warmup_epochs"],
            batch_size=self.config["transfer_learning"]["batch_size"],
            max_lr=self.config["transfer_learning"]["max_lr"],
            nce=self.config["transfer_learning"]["nce"],
            instrument=self.config["transfer_learning"]["instrument"],
        )
        rt_stats = tune_mgr.finetune_rt(transfer_lib.precursor_df)
        charge_stats = tune_mgr.finetune_charge(transfer_lib.precursor_df)
        ms2_stats = tune_mgr.finetune_ms2(
            transfer_lib.precursor_df.copy(), transfer_lib.fragment_intensity_df.copy()
        )

        tune_mgr.save_models(os.path.join(self.output_folder, self.TRANSFER_MODEL))

        combined_stats = pd.concat([rt_stats, charge_stats, ms2_stats])

        if save:
            logger.info("Writing transfer learning stats output to disk")
            write_df(
                combined_stats,
                os.path.join(self.output_folder, self.TRANSFER_STATS_OUTPUT),
                file_format="tsv",
            )

    def build_transfer_library(
        self,
        folder_list: list[str],
        keep_top: int = 3,
        number_of_processes: int = 4,
        save: bool = True,
    ) -> base.SpecLibBase:
        """
        A function to get the transfer library

        Parameters
        ----------
        folder_list : List[str]
            The list of output folders.

        keep_top : int
            The number of top runs to keep per each precursor, based on the proba. (smaller the proba, better the run)

        number_of_processes : int, optional
            The number of processes to use, by default 2

        save : bool, optional
            Whether to save the transfer library to disk, by default True

        Returns
        -------
        base.SpecLibBase
            The transfer Learning library
        """
        logger.progress("======== Building transfer library ========")
        transferAccumulator = TransferLearningAccumulator(
            keep_top=self.config["transfer_library"]["top_k_samples"],
            norm_delta_max=self.config["transfer_library"]["norm_delta_max"],
            precursor_correlation_cutoff=self.config["transfer_library"][
                "precursor_correlation_cutoff"
            ],
            fragment_correlation_ratio=self.config["transfer_library"][
                "fragment_correlation_ratio"
            ],
        )
        accumulationBroadcaster = AccumulationBroadcaster(
            folder_list=folder_list,
            number_of_processes=number_of_processes,
            processing_kwargs={
                "charged_frag_types": fragment.get_charged_frag_types(
                    self.config["transfer_library"]["fragment_types"],
                    self.config["transfer_library"]["max_charge"],
                )
            },
        )

        accumulationBroadcaster.subscribe(transferAccumulator)
        accumulationBroadcaster.run()
        logger.info(
            f"Built transfer library using {len(folder_list)} folders and {number_of_processes} processes"
        )
        log_stat_df(transfer_library_stat_df(transferAccumulator.consensus_speclibase))
        if save:
            logging.info("Writing transfer library to disk")
            transferAccumulator.consensus_speclibase.save_hdf(
                os.path.join(self.output_folder, f"{self.TRANSFER_OUTPUT}.hdf")
            )

        return transferAccumulator.consensus_speclibase

    def load_precursor_table(self):
        """Load precursor table from output folder.
        Helper functions used by other builders.

        Returns
        -------

        psm_df: pd.DataFrame
            Precursor table
        """

        return read_df(
            os.path.join(self.output_folder, f"{self.PRECURSOR_OUTPUT}"),
            file_format=self.config["search_output"]["file_format"],
        )

    def build_precursor_table(
        self,
        folder_list: list[str],
        save: bool = True,
        base_spec_lib: base.SpecLibBase = None,
    ):
        """Build precursor table from a list of seach outputs

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        save: bool
            Save the precursor table to disk

        Returns
        -------

        psm_df: pd.DataFrame
            Precursor table
        """
        logger.progress("Performing protein grouping and FDR")

        psm_df_list = []

        for folder in folder_list:
            raw_name = os.path.basename(folder)
            psm_path = os.path.join(folder, f"{self.PSM_INPUT}.parquet")

            logger.info(f"Building output for {raw_name}")

            if not os.path.exists(psm_path):
                logger.warning(f"no psm file found for {raw_name}, skipping")
            else:
                try:
                    run_df = pd.read_parquet(psm_path)
                    psm_df_list.append(run_df)
                except Exception as e:
                    logger.warning(f"Error reading psm file for {raw_name}")
                    logger.warning(e)

        if len(psm_df_list) == 0:
            raise NoPsmFoundError()

        logger.info("Building combined output")
        psm_df = pd.concat(psm_df_list)

        if base_spec_lib is not None:
            psm_df = psm_df.merge(
                base_spec_lib.precursor_df[["precursor_idx"]],
                on="precursor_idx",
                how="left",
            )

        logger.info("Performing protein inference")

        psm_df["mods"].fillna("", inplace=True)
        # make mods column a string not object
        psm_df["mods"] = psm_df["mods"].astype(str)
        psm_df["mod_sites"].fillna("", inplace=True)
        # make mod_sites column a string not object
        psm_df["mod_sites"] = psm_df["mod_sites"].astype(str)
        psm_df = precursor.hash_precursor_df(psm_df)

        if len(psm_df) == 0:
            logger.error("combined psm file is empty, can't continue")
            raise FileNotFoundError("combined psm file is empty, can't continue")

        if self.config["fdr"]["inference_strategy"] == "library":
            logger.info(
                "Inference strategy: library. Using library grouping for protein inference"
            )

            psm_df["pg"] = psm_df[self.config["fdr"]["group_level"]]
            psm_df["pg_master"] = psm_df[self.config["fdr"]["group_level"]]

        elif self.config["fdr"]["inference_strategy"] == "maximum_parsimony":
            logger.info(
                "Inference strategy: maximum_parsimony. Using maximum parsimony for protein inference"
            )

            psm_df = grouping.perform_grouping(
                psm_df, genes_or_proteins=self.config["fdr"]["group_level"], group=False
            )

        elif self.config["fdr"]["inference_strategy"] == "heuristic":
            logger.info(
                "Inference strategy: heuristic. Using maximum parsimony with grouping for protein inference"
            )

            psm_df = grouping.perform_grouping(
                psm_df, genes_or_proteins=self.config["fdr"]["group_level"], group=True
            )

        else:
            raise ValueError(
                f"Unknown inference strategy: {self.config['fdr']['inference_strategy']}. Valid options are 'library', 'maximum_parsimony' and 'heuristic'"
            )

        logger.info("Performing protein FDR")

        psm_df = perform_protein_fdr(psm_df, self._figure_path)
        psm_df = psm_df[psm_df["pg_qval"] <= self.config["fdr"]["fdr"]]

        pg_count = psm_df[psm_df["decoy"] == 0]["pg"].nunique()
        precursor_count = psm_df[psm_df["decoy"] == 0]["precursor_idx"].nunique()

        logger.progress(
            "================ Protein FDR =================",
        )
        logger.progress("Unique protein groups in output")
        logger.progress(f"  1% protein FDR: {pg_count:,}")
        logger.progress("")
        logger.progress("Unique precursor in output")
        logger.progress(f"  1% protein FDR: {precursor_count:,}")
        logger.progress(
            "================================================",
        )

        if not self.config["fdr"]["keep_decoys"]:
            psm_df = psm_df[psm_df["decoy"] == 0]
        if save:
            logger.info("Writing precursor output to disk")
            write_df(
                psm_df,
                os.path.join(self.output_folder, self.PRECURSOR_OUTPUT),
                file_format=self.config["search_output"]["file_format"],
            )

        return psm_df

    def build_stat_df(
        self,
        folder_list: list[str],
        psm_df: pd.DataFrame | None = None,
        save: bool = True,
    ):
        """Build stat table from a list of seach outputs

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        psm_df: Union[pd.DataFrame, None]
            Combined precursor table. If None, the precursor table is loaded from disk.

        save: bool
            Save the precursor table to disk

        Returns
        -------

        stat_df: pd.DataFrame
            Precursor table
        """
        logger.progress("Building search statistics")

        if self.config["search"]["channel_filter"] == "":
            all_channels = {0}
        else:
            all_channels = set(self.config["search"]["channel_filter"].split(","))

        if self.config["multiplexing"]["enabled"]:
            all_channels &= set(
                self.config["multiplexing"]["target_channels"].split(",")
            )
        all_channels = sorted([int(c) for c in all_channels])

        if psm_df is None:
            psm_df = self.load_precursor_table()
        psm_df = psm_df[psm_df["decoy"] == 0]

        stat_df_list = []
        for folder in folder_list:
            raw_name = os.path.basename(folder)
            stat_df_list.append(
                _build_run_stat_df(
                    folder,
                    raw_name,
                    psm_df[psm_df["run"] == raw_name],
                    all_channels,
                )
            )

        stat_df = pd.concat(stat_df_list)

        if save:
            logger.info("Writing stat output to disk")
            write_df(
                stat_df,
                os.path.join(self.output_folder, self.STAT_OUTPUT),
                file_format="tsv",
            )

        return stat_df

    def build_internal_df(
        self,
        folder_list: list[str],
        save: bool = True,
    ):
        """Build internal data table from a list of seach outputs

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        save: bool
            Save the precursor table to disk

        Returns
        -------

        stat_df: pd.DataFrame
            Precursor table
        """
        logger.progress("Building internal statistics")

        internal_df_list = []
        for folder in folder_list:
            internal_df_list.append(
                _build_run_internal_df(
                    folder,
                )
            )

        internal_df = pd.concat(internal_df_list)

        if save:
            logger.info("Writing internal output to disk")
            write_df(
                internal_df,
                os.path.join(self.output_folder, self.INTERNAL_OUTPUT),
                file_format="tsv",
            )

        return internal_df

    def build_lfq_tables(
        self,
        folder_list: list[str],
        psm_df: pd.DataFrame | None = None,
        save: bool = True,
    ):
        """Accumulate fragment information and perform label-free protein quantification.

        Parameters
        ----------
        folder_list: List[str]
            List of folders containing the search outputs
        psm_df: Union[pd.DataFrame, None]
            Combined precursor table. If None, the precursor table is loaded from disk.
        save: bool
            Save the precursor table to disk
        """
        logger.progress("Performing label free quantification")

        if psm_df is None:
            psm_df = self.load_precursor_table()

        # as we want to retain decoys in the output we are only removing them for lfq
        qb = QuantBuilder(psm_df[psm_df["decoy"] == 0])

        intensity_df, quality_df = qb.accumulate_frag_df_from_folders(folder_list)

        @dataclass
        class LFQOutputConfig:
            should_process: bool
            quant_level: str
            level_name: str
            save_fragments: bool = False

        quantlevel_configs = [
            LFQOutputConfig(
                self.config["search_output"]["precursor_level_lfq"],
                "mod_seq_charge_hash",
                "precursor",
                self.config["search_output"]["save_fragment_quant_matrix"],
            ),
            LFQOutputConfig(
                self.config["search_output"]["peptide_level_lfq"],
                "mod_seq_hash",
                "peptide",
            ),
            LFQOutputConfig(
                True,  # always process protein group level
                "pg",
                "pg",
            ),
        ]

        lfq_results = {}

        for quantlevel_config in quantlevel_configs:
            if not quantlevel_config.should_process:
                continue

            logger.progress(
                f"Performing label free quantification on the {quantlevel_config.level_name} level"
            )

            group_intensity_df, _ = qb.filter_frag_df(
                intensity_df,
                quality_df,
                top_n=self.config["search_output"]["min_k_fragments"],
                min_correlation=self.config["search_output"]["min_correlation"],
                group_column=quantlevel_config.quant_level,
            )

            if len(group_intensity_df) == 0:
                logger.warning(
                    f"No fragments found for {quantlevel_config.level_name}, skipping label-free quantification"
                )
                lfq_results[quantlevel_config.level_name] = pd.DataFrame()
                continue

            lfq_df = qb.lfq(
                group_intensity_df,
                quality_df,
                num_cores=self.config["general"]["thread_count"],
                min_nonan=self.config["search_output"]["min_nonnan"],
                num_samples_quadratic=self.config["search_output"][
                    "num_samples_quadratic"
                ],
                normalize=self.config["search_output"]["normalize_lfq"],
                group_column=quantlevel_config.quant_level,
            )

            lfq_results[quantlevel_config.level_name] = lfq_df

            if save:
                logger.info(f"Writing {quantlevel_config.level_name} output to disk")
                write_df(
                    lfq_df,
                    os.path.join(
                        self.output_folder, f"{quantlevel_config.level_name}.matrix"
                    ),
                    file_format=self.config["search_output"]["file_format"],
                )
                if quantlevel_config.save_fragments:
                    logger.info(
                        f"Writing fragment quantity matrix to disk, filtered on {quantlevel_config.level_name}"
                    )
                    write_df(
                        group_intensity_df,
                        os.path.join(
                            self.output_folder,
                            f"fragment_{quantlevel_config.level_name}filtered.matrix",
                        ),
                        file_format=self.config["search_output"]["file_format"],
                    )

        # Use protein group (pg) results for merging with psm_df
        pg_lfq_df = lfq_results.get("pg", pd.DataFrame())

        if len(pg_lfq_df) > 0:
            protein_df_melted = pg_lfq_df.melt(
                id_vars="pg", var_name="run", value_name="intensity"
            )
            psm_df = psm_df.merge(protein_df_melted, on=["pg", "run"], how="left")

        if save:
            logger.info("Writing psm output to disk")
            write_df(
                psm_df,
                os.path.join(self.output_folder, f"{self.PRECURSOR_OUTPUT}"),
                file_format=self.config["search_output"]["file_format"],
            )

        return lfq_results

    def build_library(
        self,
        base_spec_lib: base.SpecLibBase,
        psm_df: pd.DataFrame | None = None,
    ):
        """Build spectral library

        Parameters
        ----------

        base_spec_lib: base.SpecLibBase
            Base spectral library

        psm_df: Union[pd.DataFrame, None]
            Combined precursor table. If None, the precursor table is loaded from disk.

        """
        logger.progress("Building spectral library")

        if psm_df is None:
            psm_df = self.load_precursor_table()
        psm_df = psm_df[psm_df["decoy"] == 0]

        if len(psm_df) == 0:
            logger.warning("No precursors found, skipping library building")
            return None

        libbuilder = libtransform.MbrLibraryBuilder(
            fdr=0.01,
        )

        logger.info("Building MBR spectral library")
        mbr_spec_lib = libbuilder(psm_df, base_spec_lib)

        precursor_number = len(mbr_spec_lib.precursor_df)
        protein_number = mbr_spec_lib.precursor_df.proteins.nunique()

        logger.info(
            f"MBR spectral library contains {precursor_number:,} precursors, {protein_number:,} proteins"
        )

        if self.config["general"]["save_mbr_library"]:
            logger.info("Writing MBR spectral library to disk")
            mbr_spec_lib.save_hdf(
                os.path.join(
                    self.output_folder, f"{SearchPlanOutput.LIBRARY_OUTPUT}.hdf"
                )
            )

        return mbr_spec_lib


def _build_run_stat_df(
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
                peptidecentric.PeptideCentricWorkflow.OPTIMIZATION_MANAGER_PKL_NAME,
            )
        ):
            optimization_manager = manager.OptimizationManager(
                path=optimization_manager_path
            )
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
                peptidecentric.PeptideCentricWorkflow.CALIBRATION_MANAGER_PKL_NAME,
            )
        ):
            calibration_manager = manager.CalibrationManager(
                path=calibration_manager_path
            )

            if (
                fragment_mz_estimator := calibration_manager.get_estimator(
                    "fragment", "mz"
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
                    "precursor", "mz"
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
            stats[f"{prefix}{key}"] = calibration_stats[key]

        # collect raw stats
        raw_stats = defaultdict(lambda: np.nan)
        if os.path.exists(
            raw_file_manager_path := os.path.join(
                folder, peptidecentric.PeptideCentricWorkflow.RAW_FILE_MANAGER_PKL_NAME
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


def _get_value_or_nan(d: dict, key: str):
    try:
        return d[key]
    except KeyError:
        return np.nan


def _build_run_internal_df(
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
        folder_path, peptidecentric.PeptideCentricWorkflow.TIMING_MANAGER_PKL_NAME
    )
    raw_name = os.path.basename(folder_path)

    internal_dict = {
        "run": [raw_name],
    }

    if os.path.exists(timing_manager_path):
        timing_manager = manager.TimingManager(path=timing_manager_path)
        for key in timing_manager.timings:
            internal_dict[f"duration_{key}"] = [timing_manager.timings[key]["duration"]]

    else:
        logger.warning(f"Error reading timing manager for {raw_name}")

    return pd.DataFrame(internal_dict)


def perform_protein_fdr(psm_df: pd.DataFrame, figure_path: str) -> pd.DataFrame:
    """Perform protein FDR on PSM dataframe"""

    protein_features = []
    for _, group in psm_df.groupby(["pg", "decoy"]):
        protein_features.append(
            {
                "pg": group["pg"].iloc[0],
                "genes": group["genes"].iloc[0],
                "proteins": group["proteins"].iloc[0],
                "decoy": group["decoy"].iloc[0],
                "count": len(group),
                "n_precursor": len(group["precursor_idx"].unique()),
                "n_peptides": len(group["sequence"].unique()),
                "n_runs": len(group["run"].unique()),
                "mean_score": group["proba"].mean(),
                "best_score": group["proba"].min(),
                "worst_score": group["proba"].max(),
            }
        )

    feature_columns = [
        "count",
        "mean_score",
        "n_peptides",
        "n_precursor",
        "n_runs",
        "best_score",
        "worst_score",
    ]

    protein_features = pd.DataFrame(protein_features)

    X = protein_features[feature_columns].values
    y = protein_features["decoy"].values

    X_train, X_test, y_train, y_test = train_test_split_(
        X,
        y,
        test_size=0.2,
        random_state=42,
        exception=TooFewProteinsError,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(random_state=0).fit(X_train, y_train)

    protein_features["proba"] = clf.predict_proba(scaler.transform(X))[:, 1]
    protein_features = pd.DataFrame(protein_features)

    protein_features = fdr.get_q_values(
        protein_features,
        score_column="proba",
        decoy_column="decoy",
        qval_column="pg_qval",
    )

    n_targets = (protein_features["decoy"] == 0).sum()
    n_decoys = (protein_features["decoy"] == 1).sum()

    logger.info(
        f"Normalizing q-values using {n_targets:,} targets and {n_decoys:,} decoys"
    )

    protein_features["pg_qval"] = protein_features["pg_qval"] * n_targets / n_decoys

    if figure_path is not None:
        fdr.plot_fdr(
            X_train,
            X_test,
            y_train,
            y_test,
            clf,
            protein_features["pg_qval"],
            figure_path,
        )

    return pd.concat(
        [
            psm_df[psm_df["decoy"] == 0].merge(
                protein_features[protein_features["decoy"] == 0][["pg", "pg_qval"]],
                on="pg",
                how="left",
            ),
            psm_df[psm_df["decoy"] == 1].merge(
                protein_features[protein_features["decoy"] == 1][["pg", "pg_qval"]],
                on="pg",
                how="left",
            ),
        ]
    )


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
