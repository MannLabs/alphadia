import logging

import numpy as np
import pandas as pd

from alphadia.validation.base import Optional, Required, Schema, check_critical_values

logger = logging.getLogger()


precursors_flat_schema = Schema(
    "precursors_flat",
    [
        Required("elution_group_idx", np.uint32),
        Optional("score_group_idx", np.uint32),
        Required("precursor_idx", np.uint32),
        Required("channel", np.uint32),
        Required("decoy", np.uint8),
        Required("flat_frag_start_idx", np.uint32),
        Required("flat_frag_stop_idx", np.uint32),
        Required("charge", np.uint8),
        Required("rt_library", np.float32),
        Optional("rt_calibrated", np.float32),
        Required("mobility_library", np.float32),
        Optional("mobility_calibrated", np.float32),
        Required("mz_library", np.float32),
        Optional("mz_calibrated", np.float32),
        Required("proteins", object),
        Required("genes", object),
        *[Optional(f"i_{i}", np.float32) for i in range(10)],
    ],
)


def precursors_flat(df: pd.DataFrame, logging: bool = True):
    """Validate flat precursor dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Precursor dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------
    ValueError
        If validation fails

    """
    check_critical_values(df)
    precursors_flat_schema(df, logging=logging)


precursors_flat.__doc__ += precursors_flat_schema.docstring()

fragments_flat_schema = Schema(
    "fragments_flat",
    [
        Required("mz_library", np.float32),
        Optional("mz_calibrated", np.float32),
        Required("intensity", np.float32),
        Required("cardinality", np.uint8),
        Required("type", np.uint8),
        Required("loss_type", np.uint8),
        Required("charge", np.uint8),
        Required("number", np.uint8),
        Required("position", np.uint8),
    ],
)


def fragments_flat(df: pd.DataFrame, logging: bool = True):
    """Validate flat fragment dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Fragment dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------
    ValueError
        If validation fails

    """
    check_critical_values(df)
    fragments_flat_schema(df, logging=logging)


fragments_flat.__doc__ += fragments_flat_schema.docstring()

candidates_schema = Schema(
    "candidates_df",
    [
        Required("elution_group_idx", np.uint32),
        Required("precursor_idx", np.uint32),
        Required("rank", np.uint8),
        Required("scan_start", np.int64),
        Required("scan_stop", np.int64),
        Required("scan_center", np.int64),
        Required("frame_start", np.int64),
        Required("frame_stop", np.int64),
        Required("frame_center", np.int64),
        Optional("score", np.float32),
        Optional("score_group_idx", np.uint32),
        Optional("channel", np.uint8),
        Optional("decoy", np.uint8),
        Optional("flat_frag_start_idx", np.uint32),
        Optional("flat_frag_stop_idx", np.uint32),
        Optional("mz_library", np.float32),
        Optional("mz_calibrated", np.float32),
        *[Optional(f"i_{i}", np.float32) for i in range(10)],
    ],
)


def candidates_df(df: pd.DataFrame, logging: bool = True):
    """Validate candidate dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Candidate dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------
    ValueError
        If validation fails

    """
    check_critical_values(df)
    candidates_schema(df, logging=logging)


candidates_df.__doc__ += candidates_schema.docstring()


features_schema = Schema(
    "candidate_features_df",
    [
        Required("precursor_idx", np.uint32),
        Required("elution_group_idx", np.uint32),
        Required("rank", np.uint8),
        Required("decoy", np.uint8),
        Required("channel", np.uint8),
        Required("charge", np.uint8),
        Required("flat_frag_start_idx", np.uint32),
        Required("flat_frag_stop_idx", np.uint32),
        Required("scan_center", np.int64),
        Required("scan_start", np.int64),
        Required("scan_stop", np.int64),
        Required("frame_center", np.int64),
        Required("frame_start", np.int64),
        Required("frame_stop", np.int64),
        Required("mz_library", np.float32),
        Optional("mz_calibrated", np.float32),
        Required("mz_observed", np.float32),
        Required("rt_library", np.float32),
        Optional("rt_calibrated", np.float32),
        Required("rt_observed", np.float32),
        Required("mobility_library", np.float32),
        Optional("mobility_calibrated", np.float32),
        Required("mobility_observed", np.float32),
        *[Optional(f"i_{i}", np.float32) for i in range(10)],
    ],
)


def candidate_features_df(input_df: pd.DataFrame, logging: bool = True):
    """Validate feature dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------
    ValueError
        If validation fails

    """
    check_critical_values(input_df)
    features_schema(input_df, logging=logging)


candidate_features_df.__doc__ += features_schema.docstring()

fragment_features_schema = Schema(
    "fragment_features_df",
    [
        Required("precursor_idx", np.uint32),
        Required("rank", np.uint8),
        Required("elution_group_idx", np.uint32),
        Required("mz_library", np.float32),
        Required("mz_observed", np.float32),
        Required("mass_error", np.float32),
        Required("height", np.float32),
        Required("intensity", np.float32),
        Required("decoy", np.uint8),
    ],
)


def fragment_features_df(input_df: pd.DataFrame, logging: bool = True):
    """Validate fragment feature dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Fragment feature dataframe

    logging : bool, optional
        If True, log the validation results, by default True

    Raises
    ------
    ValueError
        If validation fails

    """
    check_critical_values(input_df)
    fragment_features_schema(input_df, logging=logging)


fragment_features_df.__doc__ += fragment_features_schema.docstring()
