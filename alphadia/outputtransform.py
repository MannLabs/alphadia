# native imports
import logging
import os

logger = logging.getLogger()

from alphadia import grouping, libtransform, utils
from alphadia import fdr

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import multiprocessing as mp

from typing import List, Tuple, Iterator, Union
import numba as nb
from alphabase.spectral_library import base

import directlfq.utils as lfqutils
import directlfq.normalization as lfqnorm
import directlfq.protein_intensity_estimation as lfqprot_estimation
import directlfq.config as lfqconfig

import logging

logger = logging.getLogger()


def get_frag_df_generator(folder_list: List[str]):
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
        frag_path = os.path.join(folder, "frag.tsv")

        if not os.path.exists(frag_path):
            logger.warning(f"no frag file found for {raw_name}")
        else:
            try:
                logger.info(f"reading frag file for {raw_name}")
                run_df = pd.read_csv(
                    frag_path,
                    sep="\t",
                    dtype={
                        "precursor_idx": np.uint32,
                        "number": np.uint8,
                        "type": np.uint8,
                    },
                )
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
        self, folder_list: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        self, df_iterable: Iterator[Tuple[str, pd.DataFrame]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        df_list = []
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
        protein_df = self.psm_df.groupby("precursor_idx", as_index=False)["pg"].first()

        intensity_df = intensity_df.merge(protein_df, on="precursor_idx", how="left")
        intensity_df.rename(columns={"pg": "protein"}, inplace=True)

        quality_df = quality_df.merge(protein_df, on="precursor_idx", how="left")
        quality_df.rename(columns={"pg": "protein"}, inplace=True)

        return intensity_df, quality_df

    def filter_frag_df(
        self,
        intensity_df: pd.DataFrame,
        quality_df: pd.DataFrame,
        min_correlation: float = 0.5,
        top_n: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            if c not in ["precursor_idx", "ion", "protein"]
        ]

        quality_df["total"] = np.mean(quality_df[run_columns].values, axis=1)
        quality_df["rank"] = quality_df.groupby("protein")["total"].rank(
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

        intensity_df.drop(columns=["precursor_idx"], inplace=True)

        lfqconfig.set_global_protein_and_ion_id(protein_id="protein", quant_id="ion")

        lfq_df = lfqutils.index_and_log_transform_input_df(intensity_df)
        lfq_df = lfqutils.remove_allnan_rows_input_df(lfq_df)
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


def prepare_df(df, psm_df, column="height"):
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
        folder_list: List[str],
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
        psm_df = self.build_precursor_table(folder_list, save=False)
        _ = self.build_stat_df(folder_list, psm_df=psm_df, save=True)
        _ = self.build_protein_table(folder_list, psm_df=psm_df, save=True)
        _ = self.build_library(base_spec_lib, psm_df=psm_df, save=True)

    def load_precursor_table(self):
        """Load precursor table from output folder.
        Helper functions used by other builders.

        Returns
        -------

        psm_df: pd.DataFrame
            Precursor table
        """

        if not os.path.exists(
            os.path.join(self.output_folder, f"{self.PRECURSOR_OUTPUT}.tsv")
        ):
            logger.error(
                f"Can't continue as no {self.PRECURSOR_OUTPUT}.tsv file was found in the output folder: {self.output_folder}"
            )
            raise FileNotFoundError(
                f"Can't continue as no {self.PRECURSOR_OUTPUT}.tsv file was found in the output folder: {self.output_folder}"
            )
        logger.info(f"Reading {self.PRECURSOR_OUTPUT}.tsv file")
        psm_df = pd.read_csv(
            os.path.join(self.output_folder, f"{self.PRECURSOR_OUTPUT}.tsv"), sep="\t"
        )
        return psm_df

    def build_precursor_table(self, folder_list: List[str], save: bool = True):
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
            psm_path = os.path.join(folder, f"{self.PSM_INPUT}.tsv")

            logger.info(f"Building output for {raw_name}")

            if not os.path.exists(psm_path):
                logger.warning(f"no psm file found for {raw_name}, skipping")
                run_df = pd.DataFrame()
            else:
                try:
                    run_df = pd.read_csv(psm_path, sep="\t")
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

        logger.info("Performing protein grouping")
        if self.config["fdr"]["library_grouping"]:
            psm_df["pg"] = psm_df[self.config["fdr"]["group_level"]]
            psm_df["pg_master"] = psm_df[self.config["fdr"]["group_level"]]
        else:
            psm_df = grouping.perform_grouping(
                psm_df, genes_or_proteins=self.config["fdr"]["group_level"]
            )

        logger.info("Performing protein FDR")
        psm_df = perform_protein_fdr(psm_df)
        psm_df = psm_df[psm_df["pg_qval"] <= self.config["fdr"]["fdr"]]

        pg_count = psm_df[psm_df["decoy"] == 0]["pg"].nunique()
        precursor_count = psm_df[psm_df["decoy"] == 0]["precursor_idx"].nunique()

        logger.progress(
            "================ Protein FDR =================",
        )
        logger.progress(f"Unique protein groups in output")
        logger.progress(f"  1% protein FDR: {pg_count:,}")
        logger.progress("")
        logger.progress(f"Unique precursor in output")
        logger.progress(f"  1% protein FDR: {precursor_count:,}")
        logger.progress(
            "================================================",
        )

        if not self.config["fdr"]["keep_decoys"]:
            psm_df = psm_df[psm_df["decoy"] == 0]

        if save:
            logger.info("Writing precursor output to disk")
            psm_df.to_csv(
                os.path.join(self.output_folder, f"{self.PRECURSOR_OUTPUT}.tsv"),
                sep="\t",
                index=False,
                float_format="%.6f",
            )

        return psm_df

    def build_stat_df(
        self,
        folder_list: List[str],
        psm_df: Union[pd.DataFrame, None] = None,
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

        if psm_df is None:
            psm_df = self.load_precursor_table()
        psm_df = psm_df[psm_df["decoy"] == 0]

        stat_df_list = []
        for folder in folder_list:
            raw_name = os.path.basename(folder)
            stat_df_list.append(
                build_stat_df(raw_name, psm_df[psm_df["run"] == raw_name])
            )

        stat_df = pd.concat(stat_df_list)

        if save:
            logger.info("Writing stat output to disk")
            stat_df.to_csv(
                os.path.join(self.output_folder, f"{self.STAT_OUTPUT}.tsv"),
                sep="\t",
                index=False,
                float_format="%.6f",
            )

        return stat_df

    def build_protein_table(
        self,
        folder_list: List[str],
        psm_df: Union[pd.DataFrame, None] = None,
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
        intensity_df, quality_df = qb.filter_frag_df(
            intensity_df,
            quality_df,
            top_n=self.config["search_output"]["min_k_fragments"],
            min_correlation=self.config["search_output"]["min_correlation"],
        )
        protein_df = qb.lfq(
            intensity_df,
            quality_df,
            num_cores=self.config["general"]["thread_count"],
            min_nonan=self.config["search_output"]["min_nonnan"],
            num_samples_quadratic=self.config["search_output"]["num_samples_quadratic"],
        )

        protein_df.rename(columns={"protein": "pg"}, inplace=True)

        protein_df_melted = protein_df.melt(
            id_vars="pg", var_name="run", value_name="intensity"
        )

        psm_df = psm_df.merge(protein_df_melted, on=["pg", "run"], how="left")

        if save:
            logger.info("Writing protein group output to disk")
            protein_df.to_csv(
                os.path.join(self.output_folder, f"{self.PG_OUTPUT}.tsv"),
                sep="\t",
                index=False,
                float_format="%.6f",
            )

            logger.info("Writing psm output to disk")
            psm_df.to_csv(
                os.path.join(self.output_folder, f"{self.PRECURSOR_OUTPUT}.tsv"),
                sep="\t",
                index=False,
                float_format="%.6f",
            )

        return protein_df

    def build_library(
        self,
        base_spec_lib: base.SpecLibBase,
        psm_df: Union[pd.DataFrame, None] = None,
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


def build_stat_df(raw_name, run_df):
    """Build stat dataframe for run"""

    base_dict = {
        "run": raw_name,
        "precursors": len(run_df),
        "proteins": run_df["pg"].nunique(),
    }

    if "weighted_mass_error" in run_df.columns:
        base_dict["ms1_accuracy"] = np.mean(run_df["weighted_mass_error"])

    if "cycle_fwhm" in run_df.columns:
        base_dict["fwhm_rt"] = np.mean(run_df["cycle_fwhm"])

    if "mobility_fwhm" in run_df.columns:
        base_dict["fwhm_mobility"] = np.mean(run_df["mobility_fwhm"])

    return pd.DataFrame(
        [
            base_dict,
        ]
    )


def perform_protein_fdr(psm_df):
    """Perform protein FDR on PSM dataframe"""

    protein_features = []
    for _, group in psm_df.groupby(["pg", "decoy"]):
        protein_features.append(
            {
                "genes": group["genes"].iloc[0],
                "proteins": group["proteins"].iloc[0],
                "decoy": group["decoy"].iloc[0],
                "count": len(group),
                "n_peptides": len(group["precursor_idx"].unique()),
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

    fdr.plot_fdr(X_train, X_test, y_train, y_test, clf, protein_features["pg_qval"])

    return pd.concat(
        [
            psm_df[psm_df["decoy"] == 0].merge(
                protein_features[protein_features["decoy"] == 0][
                    ["proteins", "pg_qval"]
                ],
                on="proteins",
                how="left",
            ),
            psm_df[psm_df["decoy"] == 1].merge(
                protein_features[protein_features["decoy"] == 1][
                    ["proteins", "pg_qval"]
                ],
                on="proteins",
                how="left",
            ),
        ]
    )
