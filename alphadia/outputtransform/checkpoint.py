"""Persist the inputs of the LFQ output step so that it can be replayed in isolation.

The checkpoint references the quant folders in place instead of copying the fragment
data, so it is only replayable on a machine that can still see those folders.
"""

import json
import logging
import os

import pandas as pd

from alphadia import __version__
from alphadia.constants.settings import LFQ_CHECKPOINT_FOLDER_NAME
from alphadia.workflow.config import Config

logger = logging.getLogger()

PSM_FILE_NAME = "psm_df.parquet"
CONFIG_FILE_NAME = "config.yaml"
METADATA_FILE_NAME = "metadata.json"
FRAG_FILE_NAME = "frag.parquet"


def save_lfq_checkpoint(
    output_folder: str,
    folder_list: list[str],
    psm_df: pd.DataFrame,
    config: Config,
) -> str:
    """Save all inputs of `SearchPlanOutput._build_lfq_tables` to disk.

    Parameters
    ----------
    output_folder: str
        Output folder of the search, the checkpoint is written to a subfolder of it

    folder_list: list[str]
        List of folders containing the search outputs

    psm_df: pd.DataFrame
        Combined precursor table

    config: Config
        Configuration object

    Returns
    -------
    str
        Path of the checkpoint folder

    """
    checkpoint_folder = os.path.join(output_folder, LFQ_CHECKPOINT_FOLDER_NAME)
    os.makedirs(checkpoint_folder, exist_ok=True)

    psm_df.to_parquet(os.path.join(checkpoint_folder, PSM_FILE_NAME), index=False)
    config.to_yaml(os.path.join(checkpoint_folder, CONFIG_FILE_NAME))

    with open(os.path.join(checkpoint_folder, METADATA_FILE_NAME), "w") as f:
        json.dump(_build_metadata(output_folder, folder_list, psm_df), f, indent=2)

    logger.info(f"Wrote LFQ checkpoint to {checkpoint_folder}")

    return checkpoint_folder


def _build_metadata(
    output_folder: str, folder_list: list[str], psm_df: pd.DataFrame
) -> dict:
    """Collect everything needed to reconstruct and size up the LFQ inputs."""
    frag_files = [_describe_frag_file(folder) for folder in folder_list]

    return {
        "alphadia_version": __version__,
        "output_folder": output_folder,
        "folder_list": folder_list,
        "psm_df": {
            "num_rows": len(psm_df),
            "columns": list(psm_df.columns),
            "memory_bytes": int(psm_df.memory_usage(deep=True).sum()),
        },
        "num_frag_files_missing": sum(not f["exists"] for f in frag_files),
        "total_frag_size_bytes": sum(f["size_bytes"] for f in frag_files),
        "frag_files": frag_files,
    }


def _describe_frag_file(folder: str) -> dict:
    """Describe the fragment file of a single quant folder without reading it."""
    frag_path = os.path.join(folder, FRAG_FILE_NAME)
    exists = os.path.exists(frag_path)

    return {
        "raw_name": os.path.basename(folder),
        "path": frag_path,
        "exists": exists,
        "size_bytes": os.path.getsize(frag_path) if exists else 0,
    }
