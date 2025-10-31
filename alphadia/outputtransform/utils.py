import logging
import os
from typing import Literal

import pandas as pd
from alphabase.peptide import precursor

from alphadia.constants.keys import (
    INTERNAL_TO_OUTPUT_MAPPING,
    InferenceStrategy,
)
from alphadia.outputtransform import grouping

logger = logging.getLogger()
supported_formats = ["parquet", "tsv"]


def read_df(path_no_format, file_format="parquet"):
    """Read dataframe from disk with choosen file format

    Parameters
    ----------

    path_no_format: str
        File to read from disk without file format

    file_format: str, default = 'parquet'
        File format for loading the file. Available options: ['parquet', 'tsv']

    Returns
    -------

    pd.DataFrame
        loaded dataframe from disk

    """

    file_path = f"{path_no_format}.{file_format}"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Can't load file as file was not found: {file_path}")

    logger.info(f"Reading {file_path} from disk")

    if file_format == "parquet":
        return pd.read_parquet(file_path)

    elif file_format == "tsv":
        return pd.read_csv(file_path, sep="\t")

    else:
        raise ValueError(
            f"Provided unknown file format: {file_format}, supported_formats: {supported_formats}"
        )


def apply_output_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert internal column names to output names and filter to only mapped columns.

    Only columns that are present in INTERNAL_TO_OUTPUT_MAPPING are kept in the output.
    This ensures that output files only contain the defined output columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with internal column names

    Returns
    -------
    pd.DataFrame
        Dataframe with output column names applied, containing only mapped columns
    """
    # Get output column names (values from the mapping)
    output_columns = set(INTERNAL_TO_OUTPUT_MAPPING.values())

    # Rename columns according to mapping
    df_renamed = df.rename(columns=INTERNAL_TO_OUTPUT_MAPPING)

    # Filter to only keep columns that are in the output mapping
    columns_to_keep = [col for col in df_renamed.columns if col in output_columns]

    return df_renamed[columns_to_keep]


def write_df(
    df: pd.DataFrame, path_no_format: str, file_format: str = "parquet"
) -> None:
    """Write dataframe from disk with chosen file format.

    Parameters
    ----------

    df: pd.DataFrame
        Dataframe to save to disk

    path_no_format: str
        Path for file without format

    file_format: str, default = 'parquet'
        File format for loading the file. Available options: ['parquet', 'tsv']

    """

    if file_format not in supported_formats:
        raise ValueError(
            f"Provided unknown file format: {file_format}, supported_formats: {supported_formats}"
        )

    file_path = f"{path_no_format}.{file_format}"

    logger.info(f"Saving {file_path} to disk")

    if file_format == "parquet":
        df.to_parquet(file_path, index=False)

    elif file_format == "tsv":
        df.to_csv(file_path, sep="\t", index=False, float_format="%.6f")


def merge_quant_levels_to_psm(
    psm_df: pd.DataFrame,
    lfq_results: dict[str, pd.DataFrame],
    quantlevel_configs: list,
) -> pd.DataFrame:
    """Merge quantification results from all levels back to the precursor table.

    Parameters
    ----------
    psm_df : pd.DataFrame
        Precursor table to merge quantification data into
    lfq_results : dict[str, pd.DataFrame]
        Dictionary containing quantification results for each level
    quantlevel_configs : list
        List of LFQOutputConfig objects defining quantification levels

    Returns
    -------
    pd.DataFrame
        Updated precursor table with merged quantification data
    """
    for config in quantlevel_configs:
        lfq_df = lfq_results.get(config.level_name)

        if lfq_df is None or lfq_df.empty:
            continue

        intensity_column = config.intensity_column

        melted_df = lfq_df.melt(
            id_vars=config.quant_level, var_name="run", value_name=intensity_column
        )
        psm_df = psm_df.merge(melted_df, on=[config.quant_level, "run"], how="left")

    return psm_df


def log_protein_fdr_summary(psm_df: pd.DataFrame) -> None:
    """Log summary statistics for protein FDR results.

    Parameters
    ----------
    psm_df : pd.DataFrame
        Precursor table with protein grouping and FDR filtering applied
    """
    pg_count = psm_df[psm_df["decoy"] == 0]["pg"].nunique()
    precursor_count = psm_df[psm_df["decoy"] == 0]["precursor_idx"].nunique()

    logger.info(
        "================ Protein FDR =================",
    )
    logger.info("Unique protein groups in output")
    logger.info(f"  1% protein FDR: {pg_count:,}")
    logger.info("")
    logger.info("Unique precursor in output")
    logger.info(f"  1% protein FDR: {precursor_count:,}")
    logger.info(
        "================================================",
    )


def load_psm_files_from_folders(
    folder_list: list[str], psm_file_name: str
) -> pd.DataFrame:
    """Load and concatenate PSM files from multiple folders.

    Parameters
    ----------
    folder_list : list[str]
        List of folders containing PSM files
    psm_file_name : str
        Name of the PSM file (without extension)

    Returns
    -------
    pd.DataFrame
        Concatenated PSM dataframe from all folders
    """
    psm_df_list = []

    for folder in folder_list:
        raw_name = os.path.basename(folder)
        psm_path = os.path.join(folder, f"{psm_file_name}.parquet")

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

    return psm_df


# TODO: remove this function in the future, shouldn't be necessary if well typed & tested
def prepare_psm_dataframe(psm_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare PSM dataframe by cleaning modification columns and hashing precursors.

    Parameters
    ----------
    psm_df : pd.DataFrame
        Raw PSM dataframe

    Returns
    -------
    pd.DataFrame
        Prepared PSM dataframe with hashed precursor information
    """
    psm_df["mods"] = psm_df["mods"].fillna("")
    psm_df["mods"] = psm_df["mods"].astype(str)
    psm_df["mod_sites"] = psm_df["mod_sites"].fillna("")
    psm_df["mod_sites"] = psm_df["mod_sites"].astype(str)
    psm_df = precursor.hash_precursor_df(psm_df)

    return psm_df


def apply_protein_inference(
    psm_df: pd.DataFrame,
    inference_strategy: Literal["library", "maximum_parsimony", "heuristic"],
    group_level: str,
) -> pd.DataFrame:
    """Apply protein inference strategy to PSM dataframe.

    Parameters
    ----------
    psm_df : pd.DataFrame
        PSM dataframe
    inference_strategy : Literal["library", "maximum_parsimony", "heuristic"]
        Inference strategy: 'library', 'maximum_parsimony', or 'heuristic'
    group_level : str
        Grouping level: 'proteins' or 'genes'

    Returns
    -------
    pd.DataFrame
        PSM dataframe with protein grouping applied
    """
    if inference_strategy == InferenceStrategy.LIBRARY:
        logger.info(
            "Inference strategy: library. Using library grouping for protein inference"
        )

        psm_df["pg"] = psm_df[group_level]
        psm_df["pg_master"] = psm_df[group_level]

    elif inference_strategy == InferenceStrategy.MAXIMUM_PARSIMONY:
        logger.info(
            "Inference strategy: maximum_parsimony. Using maximum parsimony for protein inference"
        )

        psm_df = grouping.perform_grouping(
            psm_df, genes_or_proteins=group_level, group=False
        )

    elif inference_strategy == InferenceStrategy.HEURISTIC:
        logger.info(
            "Inference strategy: heuristic. Using maximum parsimony with grouping for protein inference"
        )

        psm_df = grouping.perform_grouping(
            psm_df, genes_or_proteins=group_level, group=True
        )

    else:
        raise ValueError(
            f"Unknown inference strategy: {inference_strategy}. Valid options are {InferenceStrategy.get_values()}"
        )

    return psm_df


def get_channels_from_config(config: dict) -> list[int]:
    """Extract and compute channel list from configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing search and multiplexing settings

    Returns
    -------
    list[int]
        Sorted list of channel integers
    """
    if config["search"]["channel_filter"] == "":
        all_channels = {0}
    else:
        all_channels = set(config["search"]["channel_filter"].split(","))

    if config["multiplexing"]["enabled"]:
        all_channels &= set(config["multiplexing"]["target_channels"].split(","))

    all_channels = sorted([int(c) for c in all_channels])

    return all_channels
