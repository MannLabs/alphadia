# native imports
import logging
import os

logger = logging.getLogger()

from alphadia import grouping, libtransform
from alphadia import fdr

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class SearchPlanOutput:
    def __init__(self, config, output_folder):
        self._config = config
        self._output_folder = output_folder

    @property
    def config(self):
        return self._config

    @property
    def output_folder(self):
        return self._output_folder

    def build_output(self, folder_list, base_spec_lib):
        psm_df = self.build_precursor_table(folder_list)
        self.build_fragment_table(folder_list)
        return self.build_library(base_spec_lib, psm_df=psm_df)

    def build_precursor_table(self, folder_list):
        """Build precursor table from search plan output"""

        psm_df_list = []

        for folder in folder_list:
            raw_name = os.path.basename(folder)
            psm_path = os.path.join(folder, "psm.tsv")

            logger.progress(f"Building output for {raw_name}")

            if not os.path.exists(psm_path):
                logger.warning(f"no psm file found for {raw_name}")
                run_df = pd.DataFrame()
            else:
                try:
                    run_df = pd.read_csv(psm_path, sep="\t")
                except Exception as e:
                    logger.warning(f"Error reading psm file for {raw_name}")
                    logger.warning(e)
                    run_df = pd.DataFrame()

            psm_df_list.append(run_df)

        logger.progress("Building combined output")
        psm_df = pd.concat(psm_df_list)

        logger.progress("Performing protein grouping")
        if self.config["fdr"]["library_grouping"]:
            psm_df["pg"] = psm_df[self.config["fdr"]["group_level"]]
            psm_df["pg_master"] = psm_df[self.config["fdr"]["group_level"]]
        else:
            psm_df = grouping.perform_grouping(
                psm_df, genes_or_proteins=self.config["fdr"]["group_level"]
            )

        logger.progress("Performing protein FDR")
        psm_df = perform_protein_fdr(psm_df)
        psm_df = psm_df[psm_df["pg_qval"] <= self.config["fdr"]["fdr"]]

        print(self.config["fdr"]["keep_decoys"])
        if not self.config["fdr"]["keep_decoys"]:
            psm_df = psm_df[psm_df["decoy"] == 0]

        logger.progress("Writing combined output to disk")
        psm_df.to_csv(
            os.path.join(self.output_folder, "psm.tsv"),
            sep="\t",
            index=False,
            float_format="%.6f",
        )

        logger.progress("Building stat output")
        stat_df_list = []

        for folder in folder_list:
            raw_name = os.path.basename(folder)
            stat_df_list.append(build_stat_df(raw_name,psm_df[psm_df["run"] == raw_name]))

        stat_df = pd.concat(stat_df_list)

        logger.progress("Writing stat output to disk")
        stat_df.to_csv(
            os.path.join(self.output_folder, "stat.tsv"),
            sep="\t",
            index=False,
            float_format="%.6f",
        )

        logger.info(f"Finished building output")

        return psm_df

    def build_fragment_table(self, folder_list):
        """Build fragment table from search plan output"""
        logger.warning("Fragment table not implemented yet")

    def build_library(self, base_spec_lib, psm_df=None):
        logger.progress("Building spectral library")

        if psm_df is None:
            psm_df_path = os.path.join(self.output_folder, "psm.tsv")
            if not os.path.exists(psm_df_path):
                logger.error(
                    "Can't build MBR spectral library as no psm.tsv file was found in the output folder"
                )
                return
            logger.progress("Reading psm.tsv file")
            psm_df = pd.read_csv(
                os.path.join(self.output_folder, "psm.tsv"), sep="\t"
            )

        libbuilder = libtransform.MbrLibraryBuilder(
            fdr=0.01,
        )

        return libbuilder(
            psm_df,
            base_spec_lib
        )

  


def build_stat_df(raw_name, run_df):
    """Build stat dataframe for run"""
    
    return pd.DataFrame([{
                "run": raw_name,
                "precursors": np.sum(run_df["qval"] <= 0.01),
                "proteins": run_df[run_df["qval"] <= 0.01]["pg"].nunique(),
                "ms1_accuracy": np.mean(run_df["weighted_mass_error"]),
                "fwhm_rt": np.mean(run_df["cycle_fwhm"]),
                "fwhm_mobility": np.mean(run_df["mobility_fwhm"]),
            }
    ])


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
