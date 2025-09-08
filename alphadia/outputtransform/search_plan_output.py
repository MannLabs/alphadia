import logging
import os
from dataclasses import dataclass

import pandas as pd
from alphabase.peptide import fragment, precursor
from alphabase.spectral_library import base
from alphabase.spectral_library.base import SpecLibBase

from alphadia import utils
from alphadia.constants.keys import ConfigKeys
from alphadia.constants.settings import FIGURES_FOLDER_NAME
from alphadia.exceptions import NoPsmFoundError
from alphadia.libtransform.mbr import MbrLibraryBuilder
from alphadia.outputtransform import grouping
from alphadia.outputtransform.df_builders import (
    build_run_internal_df,
    build_run_stat_df,
    log_stat_df,
    transfer_library_stat_df,
)
from alphadia.outputtransform.outputaccumulator import (
    AccumulationBroadcaster,
    TransferLearningAccumulator,
)
from alphadia.outputtransform.protein_fdr import perform_protein_fdr
from alphadia.outputtransform.quant_builder import QuantBuilder
from alphadia.outputtransform.utils import read_df, write_df
from alphadia.transferlearning.train import FinetuneManager
from alphadia.workflow.config import Config

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

        config: Config
            Configuration object

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

    def build(self, folder_list: list[str], base_spec_lib: base.SpecLibBase | None):
        """Build output from a list of search outputs.

        The following files are written to the output folder:
        - precursor.tsv
        - protein_groups.tsv
        - stat.tsv
        - speclib.mbr.hdf

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        base_spec_lib: base.SpecLibBase, optional
            Base spectral library

        """
        logger.progress("Processing search outputs")
        psm_df = self._build_precursor_table(folder_list, save=False)
        self._build_stat_df(folder_list, psm_df=psm_df, save=True)
        self._build_internal_df(folder_list, save=True)
        self._build_lfq_tables(folder_list, psm_df=psm_df, save=True)

        if self.config["general"]["save_mbr_library"]:
            if base_spec_lib is None:
                raise ValueError(
                    "Passing base spectral library is required for MBR library building."
                )
            self._build_mbr_library(base_spec_lib, psm_df=psm_df, save=True)

        if self.config["transfer_library"]["enabled"]:
            self._build_transfer_library(folder_list, save=True)

        if self.config["transfer_learning"]["enabled"]:
            self._build_transfer_model(save=True)

    def _build_transfer_model(self, save=True):
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

    def _build_transfer_library(
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

    def _load_precursor_table(self):
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

    def _build_precursor_table(
        self,
        folder_list: list[str],
        save: bool = True,
    ):
        """Build precursor table from a list of search outputs

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

        logger.info("Building combined output")
        psm_df = pd.concat(psm_df_list)

        if len(psm_df) == 0:
            raise NoPsmFoundError()

        logger.info("Performing protein inference")

        psm_df["mods"] = psm_df["mods"].fillna("")
        # make mods column a string not object
        psm_df["mods"] = psm_df["mods"].astype(str)
        psm_df["mod_sites"] = psm_df["mod_sites"].fillna("")
        # make mod_sites column a string not object
        psm_df["mod_sites"] = psm_df["mod_sites"].astype(str)
        psm_df = precursor.hash_precursor_df(psm_df)

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

    def _build_stat_df(
        self,
        folder_list: list[str],
        psm_df: pd.DataFrame,
        save: bool = True,
    ):
        """Build stat table from a list of search outputs

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        psm_df: pd.DataFrame
            Combined precursor table

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

        psm_df = psm_df[psm_df["decoy"] == 0]

        stat_df_list = []
        for folder in folder_list:
            raw_name = os.path.basename(folder)
            stat_df_list.append(
                build_run_stat_df(
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

    def _build_internal_df(
        self,
        folder_list: list[str],
        save: bool = True,
    ):
        """Build internal data table from a list of search outputs

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
                build_run_internal_df(
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

    def _build_lfq_tables(
        self,
        folder_list: list[str],
        psm_df: pd.DataFrame,
        save: bool = True,
    ):
        """Accumulate fragment information and perform label-free protein quantification.

        Parameters
        ----------
        folder_list: List[str]
            List of folders containing the search outputs
        psm_df: pd.DataFrame
            Combined precursor table
        save: bool
            Save the precursor table to disk
        """
        logger.progress("Performing label free quantification")

        # as we want to retain decoys in the output we are only removing them for lfq
        psm_no_decoys_df = psm_df[psm_df["decoy"] == 0]
        qb = QuantBuilder(psm_no_decoys_df)

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

    def _build_mbr_library(
        self,
        base_spec_lib: base.SpecLibBase,
        psm_df: pd.DataFrame,
        save: bool = True,
    ) -> SpecLibBase | None:
        """Build MBR spectral library.

        Parameters
        ----------

        base_spec_lib: base.SpecLibBase
            Base spectral library

        psm_df: pd.DataFrame
            Combined precursor table

        save: bool
            Save the MBR spectral library to disk

        """
        logger.progress("Building MBR spectral library")

        psm_df = psm_df[psm_df["decoy"] == 0]

        if len(psm_df) == 0:
            logger.warning("No precursors found, skipping MBR library building")
            return None

        libbuilder = MbrLibraryBuilder(
            fdr=0.01,
        )
        mbr_spec_lib = libbuilder(psm_df, base_spec_lib)

        precursor_number = len(mbr_spec_lib.precursor_df)
        protein_number = mbr_spec_lib.precursor_df.proteins.nunique()

        logger.info(
            f"MBR spectral library contains {precursor_number:,} precursors, {protein_number:,} proteins"
        )

        if save:
            logger.info("Writing MBR spectral library to disk")
            mbr_spec_lib.save_hdf(
                os.path.join(
                    self.output_folder, f"{SearchPlanOutput.LIBRARY_OUTPUT}.hdf"
                )
            )

        return mbr_spec_lib
