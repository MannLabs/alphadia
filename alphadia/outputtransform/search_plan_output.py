import logging
import os

import pandas as pd
from alphabase.spectral_library import base
from alphabase.spectral_library.base import SpecLibBase

from alphadia import utils
from alphadia.constants.keys import (
    NormalizationMethods,
    SearchPlanFiles,
)
from alphadia.constants.settings import FIGURES_FOLDER_NAME
from alphadia.exceptions import NoPSMFilesFoundError, NoPSMFoundError
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
from alphadia.outputtransform.settings import OutputSettings
from alphadia.outputtransform.utils import (
    apply_output_column_names,
    apply_protein_inference,
    load_psm_files_from_folders,
    log_protein_fdr_summary,
    prepare_psm_dataframe,
    read_df,
    write_df,
    write_tsv,
)
from alphadia.transferlearning.train import FinetuneManager
from alphadia.workflow.config import Config

logger = logging.getLogger()


class SearchPlanOutput:
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
        self._settings = OutputSettings.from_config(config)
        self.output_folder = output_folder

        self._figure_path = (
            os.path.join(self.output_folder, FIGURES_FOLDER_NAME)
            if self._settings.save_figures
            else None
        )
        if self._figure_path and not os.path.exists(self._figure_path):
            os.makedirs(self._figure_path)

    def build(self, folder_list: list[str], base_spec_lib: base.SpecLibBase | None):
        """Build output from a list of search outputs.

        The following files are written to the output folder:
        - precursors
        - stat.tsv
        - internal.tsv
        - speclib.mbr.hdf, if enabled
        - speclib.transfer.hdf, peptdeep.transfer and stats.transfer.tsv, if enabled
        - one quantification matrix per level, if label-free quantification is enabled

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        base_spec_lib: base.SpecLibBase, optional
            Base spectral library

        """
        logger.progress("Processing search outputs")
        psm_df = self._build_precursor_table(folder_list)

        # every output is written as soon as it is built, so a later failure does not
        # discard the outputs already on disk
        self._write_tsv(
            self._build_stat_df(folder_list, psm_df), SearchPlanFiles.STAT_FILE_NAME
        )
        self._write_tsv(
            self._build_internal_df(folder_list), SearchPlanFiles.INTERNAL_FILE_NAME
        )

        if self._settings.save_mbr_library:
            if base_spec_lib is None:
                raise ValueError(
                    "Passing base spectral library is required for MBR library building."
                )
            mbr_spec_lib = self._build_mbr_library(base_spec_lib, psm_df)
            if mbr_spec_lib is not None:
                self._write_library(mbr_spec_lib, SearchPlanFiles.MBR_LIBRARY_FILE_NAME)

        if self._settings.transfer_library.enabled:
            self._write_library(
                self._build_transfer_library(folder_list),
                SearchPlanFiles.TRANSFER_LIBRARY_FILE_NAME,
            )

        if self._settings.transfer_learning.enabled:
            tune_mgr, transfer_stats_df = self._build_transfer_model()
            tune_mgr.save_models(
                os.path.join(
                    self.output_folder, SearchPlanFiles.TRANSFER_MODEL_FOLDER_NAME
                )
            )
            self._write_tsv(transfer_stats_df, SearchPlanFiles.TRANSFER_STATS_FILE_NAME)

        # LFQ is the most memory-intensive step and runs last, so an OOM kill here
        # does not discard the MBR library and transfer model already on disk
        # Assumption: no step upfront mutates psm_df in place
        if self._settings.lfq.normalization_method != NormalizationMethods.NONE:
            quant_output_builder = QuantOutputBuilder(psm_df, self._settings.lfq)
            lfq_results, psm_df_with_quant = quant_output_builder.build(folder_list)

            self._write_precursor_table(psm_df_with_quant)
            if lfq_results:
                quant_output_builder.save_results(
                    lfq_results,
                    self.output_folder,
                    file_format=self._settings.file_format,
                )
        else:
            logger.info("Skipping label free quantification")
            self._write_precursor_table(psm_df)

    def _write_precursor_table(self, psm_df: pd.DataFrame) -> None:
        self._write_table(
            apply_output_column_names(psm_df),
            SearchPlanFiles.PRECURSORS_FILE_STEM,
            self._settings.file_format,
        )

    def _write_table(self, df: pd.DataFrame, file_stem: str, file_format: str) -> None:
        write_df(df, os.path.join(self.output_folder, file_stem), file_format)

    def _write_tsv(self, df: pd.DataFrame, file_name: str) -> None:
        write_tsv(df, os.path.join(self.output_folder, file_name))

    def _write_library(self, spec_lib: base.SpecLibBase, file_name: str) -> None:
        library_path = os.path.join(self.output_folder, file_name)
        logger.info(f"Writing spectral library to {library_path}")
        spec_lib.save_hdf(library_path)

    def _build_transfer_model(self) -> tuple[FinetuneManager, pd.DataFrame]:
        """Finetune PeptDeep models using the transfer library on disk.

        Returns
        -------
        tuple[FinetuneManager, pd.DataFrame]
            The manager holding the finetuned models and the combined finetuning statistics
        """
        logger.progress("Train PeptDeep Models")

        transfer_lib_path = os.path.join(
            self.output_folder, SearchPlanFiles.TRANSFER_LIBRARY_FILE_NAME
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

        transfer_learning = self._settings.transfer_learning
        tune_mgr = FinetuneManager(
            device=utils.get_torch_device(transfer_learning.use_gpu),
            lr_patience=transfer_learning.lr_patience,
            test_interval=transfer_learning.test_interval,
            train_fraction=transfer_learning.train_fraction,
            validation_fraction=transfer_learning.validation_fraction,
            test_fraction=transfer_learning.test_fraction,
            epochs=transfer_learning.epochs,
            warmup_epochs=transfer_learning.warmup_epochs,
            batch_size=transfer_learning.batch_size,
            max_lr=transfer_learning.max_lr,
            nce=transfer_learning.nce,
            instrument=transfer_learning.instrument,
            charged_frag_types=self._settings.transfer_library.charged_frag_types,
        )
        rt_stats = tune_mgr.finetune_rt(transfer_lib.precursor_df)
        charge_stats = tune_mgr.finetune_charge(transfer_lib.precursor_df)
        ms2_stats = tune_mgr.finetune_ms2(
            transfer_lib.precursor_df.copy(), transfer_lib.fragment_intensity_df.copy()
        )

        return tune_mgr, pd.concat([rt_stats, charge_stats, ms2_stats])

    def _build_transfer_library(
        self,
        folder_list: list[str],
        number_of_processes: int = 4,
    ) -> base.SpecLibBase:
        """Accumulate the transfer learning library from the search outputs.

        Parameters
        ----------
        folder_list : List[str]
            The list of output folders.

        number_of_processes : int, optional
            The number of processes to use, by default 4

        Returns
        -------
        base.SpecLibBase
            The transfer Learning library
        """
        logger.progress("======== Building transfer library ========")
        transfer_library = self._settings.transfer_library
        transferAccumulator = TransferLearningAccumulator(
            keep_top=transfer_library.top_k_samples,
            norm_delta_max=transfer_library.norm_delta_max,
            precursor_correlation_cutoff=transfer_library.precursor_correlation_cutoff,
            fragment_correlation_ratio=transfer_library.fragment_correlation_ratio,
        )
        accumulationBroadcaster = AccumulationBroadcaster(
            folder_list=folder_list,
            number_of_processes=number_of_processes,
            processing_kwargs={
                "charged_frag_types": transfer_library.charged_frag_types
            },
        )

        accumulationBroadcaster.subscribe(transferAccumulator)
        accumulationBroadcaster.run()
        logger.info(
            f"Built transfer library using {len(folder_list)} folders and {number_of_processes} processes"
        )
        log_stat_df(transfer_library_stat_df(transferAccumulator.consensus_speclibase))

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
            os.path.join(self.output_folder, SearchPlanFiles.PRECURSORS_FILE_STEM),
            file_format=self._settings.file_format,
        )

    def _build_precursor_table(self, folder_list: list[str]) -> pd.DataFrame:
        """Build precursor table from a list of search outputs

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        Returns
        -------

        psm_df: pd.DataFrame
            Precursor table
        """
        logger.progress("=== Performing protein grouping and FDR ===")

        psm_df_list = load_psm_files_from_folders(folder_list)

        if len(psm_df_list) == 0:
            raise NoPSMFilesFoundError()

        psm_df = pd.concat(psm_df_list)

        if len(psm_df) == 0:
            raise NoPSMFoundError()

        logger.info("Performing protein inference")

        psm_df = prepare_psm_dataframe(psm_df)

        psm_df = apply_protein_inference(
            psm_df,
            self._settings.fdr.inference_strategy,
            self._settings.fdr.group_level,
        )

        logger.info("Performing protein FDR")

        psm_df = perform_protein_fdr(psm_df, self._figure_path)
        psm_df = psm_df[psm_df["pg_qval"] <= self._settings.fdr.fdr]

        log_protein_fdr_summary(psm_df)

        if not self._settings.fdr.keep_decoys:
            psm_df = psm_df[psm_df["decoy"] == 0]

        return psm_df

    def _build_stat_df(
        self,
        folder_list: list[str],
        psm_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build stat table from a list of search outputs

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        psm_df: pd.DataFrame
            Combined precursor table

        Returns
        -------

        stat_df: pd.DataFrame
            Stat table with one row per run and channel
        """
        logger.progress("Building search statistics")

        psm_df = psm_df[psm_df["decoy"] == 0]

        stat_df_list = []
        for folder in folder_list:
            raw_name = os.path.basename(folder)
            stat_df_list.append(
                build_run_stat_df(
                    folder,
                    raw_name,
                    psm_df[psm_df["run"] == raw_name],
                    self._settings.channels,
                )
            )

        return pd.concat(stat_df_list)

    def _build_internal_df(self, folder_list: list[str]) -> pd.DataFrame:
        """Build internal data table from a list of search outputs

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the search outputs

        Returns
        -------

        internal_df: pd.DataFrame
            Internal table with one row per run
        """
        logger.progress("Building internal statistics")

        internal_df_list = []
        for folder in folder_list:
            internal_df_list.append(
                build_run_internal_df(
                    folder,
                )
            )

        return pd.concat(internal_df_list)

    def _build_mbr_library(
        self,
        base_spec_lib: base.SpecLibBase,
        psm_df: pd.DataFrame,
    ) -> SpecLibBase | None:
        """Build MBR spectral library.

        Parameters
        ----------

        base_spec_lib: base.SpecLibBase
            Base spectral library

        psm_df: pd.DataFrame
            Combined precursor table

        Returns
        -------

        SpecLibBase | None
            MBR spectral library, None if there are no precursors
        """
        logger.progress("Building MBR spectral library")

        if len(psm_df) == 0:
            logger.warning("No precursors found, skipping MBR library building")
            return None

        libbuilder = MbrLibraryBuilder(
            fdr=0.01,
            keep_decoys=self._settings.fdr.keep_decoys_in_mbr_library,
            decoy_type=self._settings.fdr.decoy_type,
        )
        mbr_spec_lib = libbuilder(psm_df, base_spec_lib)

        precursor_number = len(mbr_spec_lib.precursor_df)
        protein_number = mbr_spec_lib.precursor_df["proteins"].nunique()

        logger.info(
            f"MBR spectral library contains {precursor_number:,} precursors, {protein_number:,} proteins"
        )

        return mbr_spec_lib
