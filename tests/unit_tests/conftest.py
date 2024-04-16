import pytest
import os
import re
from alphadia import data
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.ioff()


def mock_precursor_df(
    n_precursor: int = 100,
) -> pd.DataFrame:
    """Create a mock precursor dataframe as it's found as the individual search outputs

    Parameters
    ----------

    n_precursor : int
        Number of precursors to generate

    Returns
    -------

    precursor_df : pd.DataFrame
        A mock precursor dataframe
    """

    precursor_idx = np.arange(n_precursor)
    decoy = np.zeros(n_precursor)
    precursor_mz = np.random.rand(n_precursor) * 2000 + 500
    precursor_charge = np.random.choice([2, 3], size=n_precursor)

    proteins = np.arange(26)
    protein_names = [chr(ord("A") + i).upper() + "PROT" for i in proteins]

    proteins = np.random.choice(protein_names, size=n_precursor)
    genes = proteins

    decoy = np.concatenate([np.zeros(n_precursor // 2), np.ones(n_precursor // 2)])
    proba = np.zeros(n_precursor) + decoy * np.random.rand(n_precursor)
    qval = np.random.rand(n_precursor) * 10e-3

    random_rt = np.random.rand(n_precursor)
    random_mobility = np.random.rand(n_precursor)
    # Generate random 6 amino acid
    sequences = []
    for i in range(n_precursor):
        sequence = ""
        for j in range(6):
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
            "decoy": decoy,
            "proba": proba,
            "qval": qval,
            "sequence": sequences,
            "mods": [""] * n_precursor,
            "mod_sites": [""] * n_precursor,
        }
    )


def mock_fragment_df(n_fragments: int = 10, n_precursor: int = 10):
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

    fragment_precursor_idx = np.repeat(np.arange(n_precursor), n_fragments).reshape(
        (n_precursor, n_fragments)
    )
    fragment_mz = np.random.rand(n_precursor, n_fragments) * 200 + 2000
    fragment_charge = np.random.choice([1, 2], size=(n_precursor, n_fragments))
    fragment_charge = np.array(fragment_charge, dtype=np.uint8)
    fragment_number = np.tile(
        np.arange(1, n_fragments // 2 + 1), n_precursor * 2
    ).reshape((n_fragments, n_precursor))
    fragment_number = np.array(fragment_number, dtype=np.uint8)
    fragment_type = np.tile(np.repeat([ord("b")], n_fragments), n_precursor).reshape(
        (n_fragments, n_precursor)
    )
    fragment_type = np.array(fragment_type, dtype=np.uint8)
    fragment_height = 10 ** (precursor_intensity * 3) * np.random.rand(
        n_precursor, n_fragments
    )
    fragment_position = fragment_number - 1
    fragment_position = np.array(fragment_position, dtype=np.uint8)
    fragment_intensity = 10 ** (precursor_intensity * 3) * np.random.rand(
        n_precursor, n_fragments
    )
    fragment_correlation = np.random.rand(n_precursor, n_fragments)

    return pd.DataFrame(
        {
            "precursor_idx": fragment_precursor_idx.flatten(),
            "mz": fragment_mz.flatten(),
            "charge": fragment_charge.flatten(),
            "number": fragment_number.flatten(),
            "type": fragment_type.flatten(),
            "position": fragment_position.flatten(),
            "height": fragment_height.flatten(),
            "intensity": fragment_intensity.flatten(),
            "correlation": fragment_correlation.flatten(),
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

    # important to supress matplotlib output
