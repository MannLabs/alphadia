import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase
from conftest import mock_fragment_df, mock_precursor_df

from alphadia import outputtransform

from alphadia.outputaccumulator import ms2_quality_control
from alphadia.constants.keys import SearchStepFiles
from alphadia.workflow.base import QUANT_FOLDER_NAME


def prepare_input_data():
    """
    Create a mock precursor and fragment dataframes for 3 runs and save them in the temp_folder

    Returns
    -------
    Config : dict
        A dictionary with the configuration for the outputtransform
    temp_folder : str
        The path to the temp folder where the dataframes are saved
    raw_folders : list
        A list with the paths to the raw folders
    psm_dfs : list
        A list with the precursor dataframes (Original dataframes)
        )
    fragment_dfs : list
        A list with the fragment dataframes (Original dataframes)

    """
    run_columns = ["run_0", "run_1", "run_2"]

    config = {
        "general": {
            "thread_count": 8,
        },
        "fdr": {
            "fdr": 0.01,
            "inference_strategy": "heuristic",
            "group_level": "proteins",
            "keep_decoys": False,
        },
        "search_output": {
            "min_k_fragments": 3,
            "min_correlation": 0.25,
            "num_samples_quadratic": 50,
            "min_nonnan": 1,
            "normalize_lfq": True,
            "peptide_level_lfq": False,
            "precursor_level_lfq": False,
            "save_fragment_quant_matrix": False,
        },
        "transfer_library": {
            "enabled": True,
            "fragment_types": ["b", "y"],
            "max_charge": 2,
            "top_k_samples": 3,
            "norm_delta_max": True,
            "precursor_correlation_cutoff": 0.5,
            "fragment_correlation_ratio": 0.75,
        },
    }

    temp_folder = os.path.join(tempfile.gettempdir(), "alphadia")
    os.makedirs(temp_folder, exist_ok=True)

    quant_path = os.path.join(temp_folder, QUANT_FOLDER_NAME)
    os.makedirs(quant_path, exist_ok=True)

    # setup raw folders
    raw_folders = [os.path.join(quant_path, run) for run in run_columns]

    psm_base_df = mock_precursor_df(n_precursor=100, with_decoy=True)
    fragment_base_df = mock_fragment_df(n_precursor=200, n_fragments=10)

    psm_dfs = []
    fragment_dfs = []

    for i in range(len(run_columns)):
        psm_df = psm_base_df.sample(50)
        psm_df["run"] = run_columns[i]
        # Multiply all proba by a random constant  between  0.5 and 1.5
        psm_df["proba"] = psm_df["proba"] * np.random.rand(len(psm_df)) + 0.5
        frag_df = fragment_base_df[
            fragment_base_df["precursor_idx"].isin(psm_df["precursor_idx"])
        ]

        # sort the fragment_df, and precursors_df by precursor_idx
        psm_df = psm_df.sort_values(by="precursor_idx")
        frag_df = frag_df.sort_values(by="precursor_idx")

        # Add to the precursor_df both columns flat_frag_start_idx and flat_frag_stop_idx
        psm_df["flat_frag_start_idx"] = np.arange(0, len(psm_df) * 10, 10)
        psm_df["flat_frag_stop_idx"] = np.arange(0, len(psm_df) * 10, 10) + 9

        # Add frag_idx to the fragment_df
        frag_df["frag_idx"] = np.arange(0, len(frag_df))
        psm_dfs.append(psm_df)
        fragment_dfs.append(frag_df)

    for i, raw_folder in enumerate(raw_folders):
        os.makedirs(raw_folder, exist_ok=True)
        psm_dfs[i].to_parquet(
            os.path.join(raw_folder, SearchStepFiles.PSM_FILE_NAME), index=False
        )
        fragment_dfs[i].to_parquet(
            os.path.join(raw_folder, SearchStepFiles.FRAG_TRANSFER_FILE_NAME),
            index=False,
        )

    return config, temp_folder, raw_folders, psm_dfs, fragment_dfs


def test_complete_output_accumulation():
    """
    Test that the accumulation process is complete by making sure no unique precursor is missing

    """
    # Given:
    config, temp_folder, raw_folders, psm_dfs, fragment_dfs = prepare_input_data()
    config["transfer_library"]["top_k_samples"] = 2

    # When:
    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_transfer_library(raw_folders, save=True)
    built_lib = SpecLibBase()
    built_lib.load_hdf(
        os.path.join(temp_folder, f"{output.TRANSFER_OUTPUT}.hdf"), load_mod_seq=True
    )

    # Then: all unique none decoy precursors should be in the built library
    union_psm_df = pd.concat(psm_dfs)
    union_psm_df = union_psm_df[union_psm_df["decoy"] == 0]
    number_of_unique_precursors = len(np.unique(union_psm_df["precursor_idx"]))

    assert (
        len(np.unique(built_lib.precursor_df["precursor_idx"]))
        == number_of_unique_precursors
    ), f"{len(np.unique(built_lib.precursor_df['precursor_idx']))} != {number_of_unique_precursors}"

    shutil.rmtree(temp_folder)


def test_selection_of_precursors():
    """
    Test that the selection of precursors is done correctly by checking for
    1. No precursors with proba lower than the smallest proba in the library was not added
    2. The selected keep_top precursors are the ones with the lowest proba
    """
    # Given:
    config, temp_folder, raw_folders, psm_dfs, fragment_dfs = prepare_input_data()
    keep_top = 2
    config["transfer_library"]["top_k_samples"] = keep_top
    # When:
    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_transfer_library(raw_folders, save=True)
    built_lib = SpecLibBase()
    built_lib.load_hdf(
        os.path.join(temp_folder, f"{output.TRANSFER_OUTPUT}.hdf"), load_mod_seq=True
    )

    # Then: The selceted keep_top precursors should be the ones with the lowest proba in the original dataframes
    for precursor_idx in np.unique(built_lib.precursor_df["precursor_idx"]):
        selected_probas = built_lib.precursor_df[
            built_lib.precursor_df["precursor_idx"] == precursor_idx
        ]["proba"].values
        selected_probas = np.sort(selected_probas)
        all_probas = np.concatenate(
            [
                psm_df[psm_df["precursor_idx"] == precursor_idx]["proba"].values
                for psm_df in psm_dfs
                if precursor_idx in psm_df["precursor_idx"].values
            ]
        )
        target_kept_probas = np.sort(all_probas)[:keep_top]
        (
            np.testing.assert_almost_equal(
                selected_probas, target_kept_probas, decimal=5
            ),
            f"{selected_probas} != {target_kept_probas}",
        )

    shutil.rmtree(temp_folder)


def test_keep_top_constraint():
    """
    Test that the built library adheres to the keep top constraint by checking that the number of samples of a precursor is not more than keep_top
    """

    # Given:
    config, temp_folder, raw_folders, psm_dfs, fragment_dfs = prepare_input_data()
    keep_top = 2
    config["transfer_library"]["top_k_samples"] = keep_top

    # When:
    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_transfer_library(raw_folders, save=True)
    built_lib = SpecLibBase()
    built_lib.load_hdf(
        os.path.join(temp_folder, f"{output.TRANSFER_OUTPUT}.hdf"), load_mod_seq=True
    )

    # Then: there should be maximum keep_top precursors for each precursor
    for precursor_idx in np.unique(built_lib.precursor_df["precursor_idx"]):
        assert (
            len(
                built_lib.precursor_df[
                    built_lib.precursor_df["precursor_idx"] == precursor_idx
                ]
            )
            <= keep_top
        ), f"{len(built_lib.precursor_df[built_lib.precursor_df['precursor_idx'] == precursor_idx])} != {keep_top}"

    shutil.rmtree(temp_folder)


def test_default_column_assignment():
    """
    Test that col [rt,mobility,mz] columns are correctly assigned where:
    col = col_observed if col_observed is in columns
    col = col_calibrated if col_observed is not in columns
    col = col_library if col_observed is not in columns and col_calibrated is not in columns
    """
    # Given:
    config, temp_folder, raw_folders, psm_dfs, fragment_dfs = prepare_input_data()
    keep_top = 2
    config["transfer_library"]["top_k_samples"] = keep_top

    # When:
    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_transfer_library(raw_folders, save=True)
    built_lib = SpecLibBase()
    built_lib.load_hdf(
        os.path.join(temp_folder, f"{output.TRANSFER_OUTPUT}.hdf"), load_mod_seq=True
    )

    # Then: The columns rt, mobility, mz should be correctly assigned
    for col in ["rt", "mobility", "mz"]:
        if f"{col}_observed" in built_lib.precursor_df.columns:
            assert built_lib.precursor_df[f"{col}"].equals(
                built_lib.precursor_df[f"{col}_observed"]
            ), f"{col} != {col}_observed"
        elif f"{col}_calibrated" in built_lib.precursor_df.columns:
            assert built_lib.precursor_df[f"{col}"].equals(
                built_lib.precursor_df[f"{col}_calibrated"]
            ), f"{col} != {col}_calibrated"
        else:
            assert built_lib.precursor_df[f"{col}"].equals(
                built_lib.precursor_df[f"{col}_library"]
            ), f"{col} != {col}_library"


def test_non_nan_fragments():
    """
    Test that the accumulated fragments data frame has no nan values
    """
    # Given:
    config, temp_folder, raw_folders, psm_dfs, fragment_dfs = prepare_input_data()
    keep_top = 2
    config["transfer_library"]["top_k_samples"] = keep_top

    # When:
    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_transfer_library(raw_folders, save=True)
    built_lib = SpecLibBase()
    built_lib.load_hdf(
        os.path.join(temp_folder, f"{output.TRANSFER_OUTPUT}.hdf"), load_mod_seq=True
    )

    # Then: The fragment dataframe should have no nan values
    assert (
        not built_lib.fragment_intensity_df.isnull().values.any()
    ), "There are nan values in the fragment dataframe"


def test_use_for_ms2():
    """
    Test that the ms2 quality control is correctly applied by checking the use_for_ms2 column in the precursor_df
    """
    # Given:
    precursor_correlation_cutoff = 0.6
    fragment_correlation_ratio = 0.9

    # dummy precursor data (3 rows)
    precursor_df = pd.DataFrame(
        {
            "frag_start_idx": [0, 2, 4],
            "frag_stop_idx": [2, 4, 6],
            "sequence": ["aa", "bb", "cc"],  # Dummy sequences
        }
    )

    # Define dummy fragment intensity data (6 rows)
    fragment_intensity_df = pd.DataFrame(
        {"intensity_1": [10, 20, 30, 40, 0, 0], "intensity_2": [5, 15, 25, 35, 0, 0]}
    )

    # Define dummy fragment correlation data (6 rows)
    fragment_correlation_df = pd.DataFrame(
        {
            "intensity_1": [
                0.4,
                0.5,  # Median < 0.6 (False)
                0.8,
                0.9,  # Median > 0.6 (True)
                0.8,
                0.9,
            ],  # Median > 0.6 but all zero intensities (False)
            "intensity_2": [0.3, 0.5, 0.9, 0.7, 0.8, 0.9],
        }
    )

    speclib = SpecLibBase()
    speclib._precursor_df = precursor_df
    speclib._fragment_intensity_df = fragment_intensity_df
    speclib._fragment_correlation_df = fragment_correlation_df

    expected_use_for_ms2 = [False, True, False]
    # When:
    ms2_quality_control(
        speclib, precursor_correlation_cutoff, fragment_correlation_ratio
    )

    # Then:
    np.testing.assert_array_equal(
        speclib.precursor_df["use_for_ms2"].values, expected_use_for_ms2
    )
