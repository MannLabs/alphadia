# native imports
import logging
import os
from collections.abc import Iterator

import directlfq.config as lfqconfig
import directlfq.normalization as lfqnorm
import directlfq.protein_intensity_estimation as lfqprot_estimation
import directlfq.utils as lfqutils
import numpy as np
import pandas as pd
from alphabase.peptide import precursor
from alphabase.spectral_library import base
from alphabase.spectral_library.base import SpecLibBase
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from alphadia import fdr, grouping, libtransform, utils
from alphadia.consensus.utils import read_df, write_df
from alphadia.outputaccumulator import (
    AccumulationBroadcaster,
    TransferLearningAccumulator,
)
from alphadia.transferlearning.train import FinetuneManager

logger = logging.getLogger()


def get_frag_df_generator(folder_list: list[str]):
    """Return a generator that yields a tuple of (raw_name, frag_df)

    Parameters
    ----------

    folder_list: List[str]
        List of folders containing the frag.tsv file

    Returns
    -------

    Iterator[Tuple[str, pd.DataFrame]]
        Tuple of (raw_name, frag_df)

    """

    for folder in folder_list:
        raw_name = os.path.basename(folder)
        frag_path = os.path.join(folder, "frag.parquet")

        if not os.path.exists(frag_path):
            logger.warning(f"no frag file found for {raw_name}")
        else:
            try:
                logger.info(f"reading frag file for {raw_name}")
                run_df = pd.read_parquet(frag_path)
            except Exception as e:
                logger.warning(f"Error reading frag file for {raw_name}")
                logger.warning(e)
            else:
                yield raw_name, run_df


class QuantBuilder:
    def __init__(self, psm_df, column="intensity"):
        self.psm_df = psm_df
        self.column = column

    def accumulate_frag_df_from_folders(
        self, folder_list: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Accumulate the fragment data from a list of folders

        Parameters
        ----------

        folder_list: List[str]
            List of folders containing the frag.tsv file

        Returns
        -------
        intensity_df: pd.DataFrame
            Dataframe with the intensity data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        quality_df: pd.DataFrame
            Dataframe with the quality data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...
        """

        df_iterable = get_frag_df_generator(folder_list)
        return self.accumulate_frag_df(df_iterable)

    def accumulate_frag_df(
        self, df_iterable: Iterator[tuple[str, pd.DataFrame]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Consume a generator of (raw_name, frag_df) tuples and accumulate the data in a single dataframe

        Parameters
        ----------

        df_iterable: Iterator[Tuple[str, pd.DataFrame]]
            Iterator of (raw_name, frag_df) tuples

        Returns
        -------
        intensity_df: pd.DataFrame
            Dataframe with the intensity data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        quality_df: pd.DataFrame
            Dataframe with the quality data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...
        """

        logger.info("Accumulating fragment data")

        raw_name, df = next(df_iterable, (None, None))
        if df is None:
            logger.warning(f"no frag file found for {raw_name}")
            return

        df = prepare_df(df, self.psm_df, column=self.column)

        intensity_df = df[["precursor_idx", "ion", self.column]].copy()
        intensity_df.rename(columns={self.column: raw_name}, inplace=True)

        quality_df = df[["precursor_idx", "ion", "correlation"]].copy()
        quality_df.rename(columns={"correlation": raw_name}, inplace=True)

        for raw_name, df in df_iterable:
            df = prepare_df(df, self.psm_df, column=self.column)

            intensity_df = intensity_df.merge(
                df[["ion", self.column, "precursor_idx"]],
                on=["ion", "precursor_idx"],
                how="outer",
            )
            intensity_df.rename(columns={self.column: raw_name}, inplace=True)

            quality_df = quality_df.merge(
                df[["ion", "correlation", "precursor_idx"]],
                on=["ion", "precursor_idx"],
                how="outer",
            )
            quality_df.rename(columns={"correlation": raw_name}, inplace=True)

        # replace nan with 0
        intensity_df.fillna(0, inplace=True)
        quality_df.fillna(0, inplace=True)

        intensity_df["precursor_idx"] = intensity_df["precursor_idx"].astype(np.uint32)
        quality_df["precursor_idx"] = quality_df["precursor_idx"].astype(np.uint32)

        # annotate protein group
        annotate_df = self.psm_df.groupby("precursor_idx", as_index=False).agg(
            {"pg": "first", "mod_seq_hash": "first", "mod_seq_charge_hash": "first"}
        )

        intensity_df = intensity_df.merge(annotate_df, on="precursor_idx", how="left")
        quality_df = quality_df.merge(annotate_df, on="precursor_idx", how="left")

        return intensity_df, quality_df

    def filter_frag_df(
        self,
        intensity_df: pd.DataFrame,
        quality_df: pd.DataFrame,
        min_correlation: float = 0.5,
        top_n: int = 3,
        group_column: str = "pg",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter the fragment data by quality

        Parameters
        ----------
        intensity_df: pd.DataFrame
            Dataframe with the intensity data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        quality_df: pd.DataFrame
            Dataframe with the quality data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        min_correlation: float
            Minimum correlation to keep a fragment, if not below top_n

        top_n: int
            Keep the top n fragments per precursor

        Returns
        -------

        intensity_df: pd.DataFrame
            Dataframe with the intensity data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        quality_df: pd.DataFrame
            Dataframe with the quality data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        """

        logger.info("Filtering fragments by quality")

        run_columns = [
            c
            for c in intensity_df.columns
            if c
            not in ["precursor_idx", "ion", "pg", "mod_seq_hash", "mod_seq_charge_hash"]
        ]

        quality_df["total"] = np.mean(quality_df[run_columns].values, axis=1)
        quality_df["rank"] = quality_df.groupby(group_column)["total"].rank(
            ascending=False, method="first"
        )
        mask = (quality_df["rank"].values <= top_n) | (
            quality_df["total"].values > min_correlation
        )
        return intensity_df[mask], quality_df[mask]

    def lfq(
        self,
        intensity_df: pd.DataFrame,
        quality_df: pd.DataFrame,
        num_samples_quadratic: int = 50,
        min_nonan: int = 1,
        num_cores: int = 8,
        normalize: bool = True,
        group_column: str = "pg",
    ) -> pd.DataFrame:
        """Perform label-free quantification

        Parameters
        ----------

        intensity_df: pd.DataFrame
            Dataframe with the intensity data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        quality_df: pd.DataFrame
            Dataframe with the quality data containing the columns precursor_idx, ion, raw_name1, raw_name2, ...

        Returns
        -------

        lfq_df: pd.DataFrame
            Dataframe with the label-free quantification data containing the columns precursor_idx, ion, intensity, protein

        """

        logger.info("Performing label-free quantification using directLFQ")

        # drop all other columns as they will be interpreted as samples
        columns_to_drop = list(
            {"precursor_idx", "pg", "mod_seq_hash", "mod_seq_charge_hash"}
            - {group_column}
        )
        _intensity_df = intensity_df.drop(columns=columns_to_drop)

        lfqconfig.set_global_protein_and_ion_id(protein_id=group_column, quant_id="ion")
        lfqconfig.set_compile_normalized_ion_table(
            compile_normalized_ion_table=False
        )  # save compute time by avoiding the creation of a normalized ion table
        lfqconfig.check_wether_to_copy_numpy_arrays_derived_from_pandas()  # avoid read-only pandas bug on linux if applicable
        lfqconfig.set_log_processed_proteins(
            log_processed_proteins=True
        )  # here you can chose wether to log the processed proteins or not

        _intensity_df.sort_values(by=group_column, inplace=True, ignore_index=True)

        lfq_df = lfqutils.index_and_log_transform_input_df(_intensity_df)
        lfq_df = lfqutils.remove_allnan_rows_input_df(lfq_df)

        if normalize:
            lfq_df = lfqnorm.NormalizationManagerSamplesOnSelectedProteins(
                lfq_df,
                num_samples_quadratic=num_samples_quadratic,
                selected_proteins_file=None,
            ).complete_dataframe

        protein_df, _ = lfqprot_estimation.estimate_protein_intensities(
            lfq_df,
            min_nonan=min_nonan,
            num_samples_quadratic=num_samples_quadratic,
            num_cores=num_cores,
        )

        return protein_df


def prepare_df(df, psm_df, column="intensity"):
    df = df[df["precursor_idx"].isin(psm_df["precursor_idx"])].copy()
    df["ion"] = utils.ion_hash(
        df["precursor_idx"].values,
        df["number"].values,
        df["type"].values,
        df["charge"].values,
    )
    return df[["precursor_idx", "ion", column, "correlation"]]


class SearchPlanOutput:
    PSM_INPUT = "psm"
    PRECURSOR_OUTPUT = "precursors"
    STAT_OUTPUT = "stat"
    PG_OUTPUT = "protein_groups"
    LIBRARY_OUTPUT = "speclib.mbr"
    TRANSFER_OUTPUT = "speclib.transfer"
    TRANSFER_MODEL = "peptdeep.transfer"

    def __init__(self, config: dict, output_folder: str):
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
        self._config = config
        self._output_folder = output_folder

    @property
    def config(self):
        return self._config

    @property
    def output_folder(self):
        return self._output_folder

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
        _ = self.build_lfq_tables(folder_list, psm_df=psm_df, save=True)
        _ = self.build_library(base_spec_lib, psm_df=psm_df, save=True)

        if self.config["transfer_library"]["enabled"]:
            _ = self.build_transfer_library(folder_list, save=True)

        if self.config["transfer_learning"]["enabled"]:
            _ = self.build_transfer_model()

    def build_transfer_model(self):
        logger.progress("Train PeptDeep Models")

        transfer_lib_path = os.path.join(
            self.output_folder, f"{self.TRANSFER_OUTPUT}.hdf"
        )
        assert os.path.exists(
            transfer_lib_path
        ), f"Transfer library not found at {transfer_lib_path}, did you enable library generation?"

        transfer_lib = SpecLibBase()
        transfer_lib.load_hdf(
            transfer_lib_path,
            load_mod_seq=True,
        )

        device = utils.get_torch_device(self.config["general"]["use_gpu"])

        tune_mgr = FinetuneManager(
            device=device, settings=self.config["transfer_learning"]
        )
        tune_mgr.finetune_rt(transfer_lib.precursor_df)
        tune_mgr.finetune_charge(transfer_lib.precursor_df)
        tune_mgr.finetune_ms2(
            transfer_lib.precursor_df.copy(), transfer_lib.fragment_intensity_df.copy()
        )

        tune_mgr.save_models(os.path.join(self.output_folder, self.TRANSFER_MODEL))

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
            folder_list, number_of_processes
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
                run_df = pd.DataFrame()
            else:
                try:
                    run_df = pd.read_parquet(psm_path)
                except Exception as e:
                    logger.warning(f"Error reading psm file for {raw_name}")
                    logger.warning(e)
                    run_df = pd.DataFrame()

            psm_df_list.append(run_df)

        if len(psm_df_list) == 0:
            logger.error("No psm files found, can't continue")
            raise FileNotFoundError("No psm files found, can't continue")

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
        psm_df = perform_protein_fdr(psm_df)
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
                    raw_name, psm_df[psm_df["run"] == raw_name], all_channels
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

        group_list = []
        group_nice_list = []

        if self.config["search_output"]["peptide_level_lfq"]:
            group_list.append("mod_seq_hash")
            group_nice_list.append("peptide")

        if self.config["search_output"]["precursor_level_lfq"]:
            group_list.append("mod_seq_charge_hash")
            group_nice_list.append("precursor")

        group_list.append("pg")
        group_nice_list.append("pg")

        # IMPORTANT: 'pg' has to be the last group in the list as this will be reused
        for group, group_nice in zip(group_list, group_nice_list, strict=True):
            logger.progress(
                f"Performing label free quantification on the {group_nice} level"
            )

            group_intensity_df, _ = qb.filter_frag_df(
                intensity_df,
                quality_df,
                top_n=self.config["search_output"]["min_k_fragments"],
                min_correlation=self.config["search_output"]["min_correlation"],
                group_column=group,
            )

            lfq_df = qb.lfq(
                group_intensity_df,
                quality_df,
                num_cores=self.config["general"]["thread_count"],
                min_nonan=self.config["search_output"]["min_nonnan"],
                num_samples_quadratic=self.config["search_output"][
                    "num_samples_quadratic"
                ],
                normalize=self.config["search_output"]["normalize_lfq"],
                group_column=group,
            )

            if save:
                logger.info(f"Writing {group_nice} output to disk")

                write_df(
                    lfq_df,
                    os.path.join(self.output_folder, f"{group_nice}.matrix"),
                    file_format=self.config["search_output"]["file_format"],
                )

        protein_df_melted = lfq_df.melt(
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

        return lfq_df

    def build_library(
        self,
        base_spec_lib: base.SpecLibBase,
        psm_df: pd.DataFrame | None = None,
        save: bool = True,
    ):
        """Build spectral library

        Parameters
        ----------

        base_spec_lib: base.SpecLibBase
            Base spectral library

        psm_df: Union[pd.DataFrame, None]
            Combined precursor table. If None, the precursor table is loaded from disk.

        save: bool
            Save the generated spectral library to disk

        """
        logger.progress("Building spectral library")

        if psm_df is None:
            psm_df = self.load_precursor_table()
        psm_df = psm_df[psm_df["decoy"] == 0]

        libbuilder = libtransform.MbrLibraryBuilder(
            fdr=0.01,
        )

        logger.info("Building MBR spectral library")
        mbr_spec_lib = libbuilder(psm_df, base_spec_lib)

        precursor_number = len(mbr_spec_lib.precursor_df)
        protein_number = mbr_spec_lib.precursor_df.proteins.nunique()

        # use comma to separate thousands
        logger.info(
            f"MBR spectral library contains {precursor_number:,} precursors, {protein_number:,} proteins"
        )

        logger.info("Writing MBR spectral library to disk")
        mbr_spec_lib.save_hdf(os.path.join(self.output_folder, "speclib.mbr.hdf"))

        if save:
            logger.info("Writing MBR spectral library to disk")
            mbr_spec_lib.save_hdf(os.path.join(self.output_folder, "speclib.mbr.hdf"))

        return mbr_spec_lib


def _build_run_stat_df(
    raw_name: str, run_df: pd.DataFrame, channels: list[int] | None = None
):
    """Build stat dataframe for a single run.

    Parameters
    ----------

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
    out_df = []

    for channel in channels:
        channel_df = run_df[run_df["channel"] == channel]

        base_dict = {
            "run": raw_name,
            "channel": channel,
            "precursors": len(channel_df),
            "proteins": channel_df["pg"].nunique(),
        }

        if "weighted_mass_error" in channel_df.columns:
            base_dict["ms1_accuracy"] = np.mean(channel_df["weighted_mass_error"])

        if "cycle_fwhm" in channel_df.columns:
            base_dict["fwhm_rt"] = np.mean(channel_df["cycle_fwhm"])

        if "mobility_fwhm" in channel_df.columns:
            base_dict["fwhm_mobility"] = np.mean(channel_df["mobility_fwhm"])

        out_df.append(base_dict)

    return pd.DataFrame(out_df)


def perform_protein_fdr(psm_df):
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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

    fdr.plot_fdr(X_train, X_test, y_train, y_test, clf, protein_features["pg_qval"])

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
