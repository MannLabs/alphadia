import os
import tempfile
import numpy as np
from conftest import _mock_precursor_df, _mock_fragment_df
from alphadia import outputtransform
from alphabase.spectral_library.base import SpecLibBase
import shutil


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
        },
    }

    temp_folder = os.path.join(tempfile.gettempdir(), "alphadia")
    os.makedirs(temp_folder, exist_ok=True)

    progress_folder = os.path.join(temp_folder, "progress")
    os.makedirs(progress_folder, exist_ok=True)

    # setup raw folders
    raw_folders = [os.path.join(progress_folder, run) for run in run_columns]

    psm_base_df = _mock_precursor_df(n_precursor=100)
    fragment_base_df = _mock_fragment_df(n_precursor=200, n_fragments=10)

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
        psm_dfs[i].to_csv(os.path.join(raw_folder, "psm.tsv"), sep="\t", index=False)
        fragment_dfs[i].to_csv(
            os.path.join(raw_folder, "frag.tsv"), sep="\t", index=False
        )

    return config, temp_folder, raw_folders, psm_dfs, fragment_dfs


def test_complete_output_accumulation():
    """
    Test that the accumulation process is complete by making sure no unique precursor is missing

    """
    # Given:
    config, temp_folder, raw_folders, psm_dfs, fragment_dfs = prepare_input_data()
    keep_top = 2
    # When:
    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_transfer_library(raw_folders, keep_top=keep_top, save=True)
    built_lib = SpecLibBase()
    built_lib.load_hdf(
        os.path.join(temp_folder, f"{output.TRANSFER_OUTPUT}.hdf"), load_mod_seq=True
    )

    # Then: all unique precursors should be in the built library
    number_of_unique_precursors = len(
        np.unique(
            np.concatenate([psm_df["precursor_idx"].values for psm_df in psm_dfs])
        )
    )
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
    # When:
    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_transfer_library(raw_folders, keep_top=keep_top, save=True)
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

    # When:
    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_transfer_library(raw_folders, keep_top=keep_top, save=True)
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
