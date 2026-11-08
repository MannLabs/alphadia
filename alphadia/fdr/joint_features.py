"""Per-run files of the first-pass final FDR round that the joint rescoring over all runs reads.

Each run writes every final-round row (after the riding removal, before fragment competition) with its classifier
features as float32, the identity columns, and its cross-fitted per-run probability; after the fragment competition it
writes which of those rows lost their fragments.
"""

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

JOINT_FEATURES_FILE_NAME = "joint_features.parquet"
JOINT_LOST_FILE_NAME = "joint_fragcomp_lost.npy"
# the classifier feature names, stored in the parquet schema: naa and rt_observed are features and identity columns
FEATURE_COLUMNS_METADATA_KEY = b"alphadia_joint_feature_columns"
ID_COLUMNS = ["precursor_idx", "elution_group_idx", "channel", "decoy", "rank", "naa"]
PROBA_COLUMN = "proba"
WRITE_CHUNK_ROWS = 1_000_000


def features_path(folder: str) -> str:
    return os.path.join(folder, JOINT_FEATURES_FILE_NAME)


def lost_path(folder: str) -> str:
    return os.path.join(folder, JOINT_LOST_FILE_NAME)


def write_run_features(
    folder: str,
    psm_df: pd.DataFrame,
    X: np.ndarray,  # noqa: N803
    x_columns: list[str],
    proba: np.ndarray,
) -> None:
    """Write the rows of `X` (row-aligned with `psm_df` and `proba`) to the run's joint feature file, in chunks so
    that no second copy of the matrix is held."""
    ids = {
        column: psm_df[column].to_numpy()
        for column in ID_COLUMNS
        if column in psm_df.columns and column not in x_columns
    }
    schema = pa.schema(
        [pa.field(column, pa.float32()) for column in x_columns]
        + [pa.field(column, pa.from_numpy_dtype(v.dtype)) for column, v in ids.items()]
        + [pa.field(PROBA_COLUMN, pa.float64())],
        metadata={FEATURE_COLUMNS_METADATA_KEY: json.dumps(list(x_columns))},
    )
    with pq.ParquetWriter(features_path(folder), schema) as writer:
        for start in range(0, len(X), WRITE_CHUNK_ROWS):
            stop = start + WRITE_CHUNK_ROWS
            arrays = (
                [
                    pa.array(X[start:stop, j].astype(np.float32))
                    for j in range(len(x_columns))
                ]
                + [pa.array(v[start:stop]) for v in ids.values()]
                + [pa.array(np.asarray(proba[start:stop], dtype=np.float64))]
            )
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))


def write_fragcomp_lost(folder: str, lost: np.ndarray) -> None:
    np.save(lost_path(folder), np.asarray(lost, dtype=bool))


def has_run_features(folder: str) -> bool:
    return os.path.exists(features_path(folder))


def feature_columns(folder: str) -> list[str]:
    metadata = pq.ParquetFile(features_path(folder)).schema_arrow.metadata
    return json.loads(metadata[FEATURE_COLUMNS_METADATA_KEY])


def read_ids(folder: str) -> pd.DataFrame:
    """The identity columns and the per-run probability of every row, and `lost` (True for the rows that lost the
    run's fragment competition)."""
    path = features_path(folder)
    names = pq.ParquetFile(path).schema_arrow.names
    ids = pq.read_table(
        path, columns=[c for c in [*ID_COLUMNS, PROBA_COLUMN] if c in names]
    ).to_pandas()
    ids["lost"] = (
        np.load(lost_path(folder))
        if os.path.exists(lost_path(folder))
        else np.zeros(len(ids), dtype=bool)
    )
    return ids


def read_features(
    folder: str, columns: list[str], rows: np.ndarray | None = None
) -> np.ndarray:
    """The float32 feature matrix of `columns` for `rows` (all rows if None), read one column at a time: a pandas
    frame of a whole file would cost several times the matrix."""
    path = features_path(folder)
    n_rows = pq.ParquetFile(path).metadata.num_rows if rows is None else len(rows)
    x = np.empty((n_rows, len(columns)), dtype=np.float32)
    for j, column in enumerate(columns):
        values = pq.read_table(path, columns=[column]).column(0).to_numpy()
        x[:, j] = values if rows is None else values[rows]
    return x


def remove_run_files(folder: str) -> None:
    for path in (features_path(folder), lost_path(folder)):
        if os.path.exists(path):
            os.remove(path)
