import os
import re
import tempfile

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.ioff()


def mock_precursor_df(
    n_precursor: int = 100,
    with_decoy=True,
) -> pd.DataFrame:
    """Create a mock precursor dataframe as it's found as the individual search outputs

    Parameters
    ----------

    n_precursor : int
        Number of precursors to generate

    with_decoy : bool
        If True, half of the precursors will be decoys

    Returns
    -------

    precursor_df : pd.DataFrame
        A mock precursor dataframe
    """

    precursor_idx = np.arange(n_precursor)
    precursor_mz = np.random.rand(n_precursor) * 2000 + 500
    precursor_charge = np.random.choice([2, 3], size=n_precursor)

    proteins = np.arange(26)
    protein_names = [chr(ord("A") + i).upper() + "PROT" for i in proteins]

    proteins = np.random.choice(protein_names, size=n_precursor)
    genes = proteins

    if with_decoy:
        decoy = np.concatenate([np.zeros(n_precursor // 2), np.ones(n_precursor // 2)])
    else:
        decoy = np.zeros(n_precursor)
    proba = np.zeros(n_precursor) + decoy * np.random.rand(n_precursor)
    qval = np.random.rand(n_precursor) * 10e-3

    random_rt = np.random.rand(n_precursor)
    random_mobility = np.random.rand(n_precursor)
    # Generate random 6 amino acid
    sequences = []
    for _ in range(n_precursor):
        sequence = ""
        for __ in range(6):
            sequence += chr(np.random.randint(65, 91))
        sequences.append(sequence)
    return pd.DataFrame(
        {
            "precursor_idx": precursor_idx,
            "decoy": decoy,
            "mz_library": precursor_mz,
            "rt_library": random_rt,
            "mobility_library": random_mobility,
            "mz_observed": precursor_mz,
            "rt_observed": random_rt,
            "mobility_observed": random_mobility,
            "mz_calibrated": precursor_mz,
            "rt_calibrated": random_rt,
            "charge": precursor_charge,
            "proteins": proteins,
            "genes": genes,
            "proba": proba,
            "qval": qval,
            "sequence": sequences,
            "mods": [""] * n_precursor,
            "mod_sites": [""] * n_precursor,
            "channel": [0] * n_precursor,
        }
    )


def mock_fragment_df(n_fragments: int = 10, n_precursor: int = 20):
    """Create a mock fragment dataframe as it's found as the individual search outputs

    Parameters
    ----------

    n_fragments : int
        Number of fragments per precursor

    n_precursor : int
        Number of precursors to generate

    Returns
    -------

    fragment_df : pd.DataFrame
        A mock fragment dataframe
    """

    precursor_intensity = np.random.rand(n_precursor, 1)

    # create a dataframe with n_precursor * n_fragments rows

    # each column is a list of n_precursor * n_fragments elements
    fragment_precursor_idx = np.repeat(np.arange(n_precursor), n_fragments).flatten()
    fragment_mz = np.random.rand(n_precursor * n_fragments) * 2000 + 200
    fragment_charge = (
        np.random.choice([1, 2], size=(n_precursor * n_fragments))
        .astype(np.uint8)
        .flatten()
    )

    fragment_number = (
        np.tile(
            np.concatenate(
                [np.arange(1, n_fragments // 2 + 1), np.arange(10 // 2, 0, -1)]
            ),
            n_precursor,
        )
        .astype(np.uint8)
        .flatten()
    )

    fragment_type = (
        np.tile(np.repeat([ord("b"), ord("y")], n_fragments // 2), n_precursor)
        .astype(np.uint8)
        .flatten()
    )

    fragment_height = (
        10 ** (precursor_intensity * 3) * np.random.rand(n_fragments)
    ).flatten()

    fragment_position = (
        np.tile(np.arange(0, n_fragments // 2), n_precursor * 2)
        .astype(np.uint8)
        .flatten()
    )

    fragment_intensity = (
        10 ** (precursor_intensity * 3) * np.random.rand(n_fragments)
    ).flatten()
    fragment_correlation = np.random.rand(n_precursor * n_fragments).flatten()

    return pd.DataFrame(
        {
            "precursor_idx": fragment_precursor_idx,
            "mz": fragment_mz,
            "charge": fragment_charge,
            "number": fragment_number,
            "type": fragment_type,
            "position": fragment_position,
            "height": fragment_height,
            "intensity": fragment_intensity,
            "correlation": fragment_correlation,
        }
    )


def pytest_configure(config):
    test_data_path = os.environ.get("TEST_DATA_DIR", None)

    pytest.test_data = {}

    if test_data_path is None:
        return

    # get all folders in the test_data_path
    raw_folders = [
        item
        for item in os.listdir(test_data_path)
        if os.path.isdir(os.path.join(test_data_path, item))
    ]

    for raw_folder in raw_folders:
        raw_files = [
            os.path.join(test_data_path, raw_folder, item)
            for item in os.listdir(os.path.join(test_data_path, raw_folder))
            if bool(re.search("(.d|.raw|.hdf)$", item))
        ]
        pytest.test_data[raw_folder] = raw_files

    # set numba environment variables
    os.environ["NUMBA_BOUNDSCHECK"] = "1"
    os.environ["NUMBA_DEVELOPER_MODE"] = "1"
    os.environ["NUMBA_FULL_TRACEBACKS"] = "1"


def random_tempfolder():
    """Create a randomly named temp folder in the system temp folder

    Returns
    -------
    path : str
        Path to the created temp folder

    """
    tempdir = tempfile.gettempdir()
    # 6 alphanumeric characters
    random_foldername = "alphadia_" + "".join(
        np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 6)
    )
    path = os.path.join(tempdir, random_foldername)
    os.makedirs(path, exist_ok=True)
    print(f"Created temp folder: {path}")
    return path
