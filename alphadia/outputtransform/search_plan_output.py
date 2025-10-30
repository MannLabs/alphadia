import logging
import os

import pandas as pd
from alphabase.peptide import fragment
from alphabase.spectral_library import base
from alphabase.spectral_library.base import SpecLibBase

from alphadia import utils
from alphadia.constants.keys import (
    ConfigKeys,
)
from alphadia.constants.settings import FIGURES_FOLDER_NAME
from alphadia.exceptions import NoPsmFoundError
from alphadia.libtransform.mbr import MbrLibraryBuilder
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
from alphadia.outputtransform.quantification import QuantOutputBuilder
from alphadia.outputtransform.utils import (
    apply_output_column_names,
    apply_protein_inference,
    get_channels_from_config,
    load_psm_files_from_folders,
    log_protein_fdr_summary,
    prepare_psm_dataframe,
    read_df,
    write_df,
)
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

        psm_df = load_psm_files_from_folders(folder_list, self.PSM_INPUT)

        if len(psm_df) == 0:
            raise NoPsmFoundError()

        logger.info("Performing protein inference")

        psm_df = prepare_psm_dataframe(psm_df)

        psm_df = apply_protein_inference(
            psm_df,
            self.config["fdr"]["inference_strategy"],
            self.config["fdr"]["group_level"],
        )

        logger.info("Performing protein FDR")

        psm_df = perform_protein_fdr(psm_df, self._figure_path)
        psm_df = psm_df[psm_df["pg_qval"] <= self.config["fdr"]["fdr"]]

        log_protein_fdr_summary(psm_df)

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

        all_channels = get_channels_from_config(self.config)

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
        quant_output_builder = QuantOutputBuilder(psm_df, self.config)
        lfq_results, psm_df_with_quant = quant_output_builder.build(folder_list)

        if save and lfq_results:
            quant_output_builder.save_results(
                lfq_results,
                self.output_folder,
                file_format=self.config["search_output"]["file_format"],
            )

            logger.info("Writing psm output to disk")
            psm_df_output = apply_output_column_names(psm_df_with_quant)
            write_df(
                psm_df_output,
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
