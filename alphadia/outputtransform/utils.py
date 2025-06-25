import logging
import os

import pandas as pd

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


def write_df(df, path_no_format, file_format="parquet"):
    """Read dataframe from disk with choosen file format

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

    else:
        raise ValueError("I don't know how you ended up here")
