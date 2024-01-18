import tempfile
from alphadia import outputtransform
import pandas as pd
import numpy as np
import os
import shutil


def _mock_precursor_df(
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

    return pd.DataFrame(
        {
            "precursor_idx": precursor_idx,
            "decoy": decoy,
            "mz_library": precursor_mz,
            "charge": precursor_charge,
            "proteins": proteins,
            "genes": genes,
            "decoy": decoy,
            "proba": proba,
            "qval": qval,
            "sequence": ["AAAAAA"] * n_precursor,
            "mods": [""] * n_precursor,
            "mod_sites": [""] * n_precursor,
        }
    )


_mock_precursor_df()


def _mock_fragment_df(n_fragments: int = 10, n_precursor: int = 10):
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
    fragment_number = np.tile(np.arange(n_fragments // 2), n_precursor * 2).reshape(
        (n_fragments, n_precursor)
    )
    fragment_type = np.tile(
        np.repeat([ord("b"), ord("y")], n_fragments // 2), n_precursor
    ).reshape((n_fragments, n_precursor))

    fragment_height = 10 ** (precursor_intensity * 3) * np.random.rand(
        n_precursor, n_fragments
    )
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
            "height": fragment_height.flatten(),
            "intensity": fragment_intensity.flatten(),
            "correlation": fragment_correlation.flatten(),
        }
    )


def test_output_transform():
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
    fragment_base_df = _mock_fragment_df(n_precursor=200)

    for raw_folder in raw_folders:
        os.makedirs(raw_folder, exist_ok=True)

        psm_df = psm_base_df.sample(50)
        psm_df["run"] = os.path.basename(raw_folder)
        frag_df = fragment_base_df[
            fragment_base_df["precursor_idx"].isin(psm_df["precursor_idx"])
        ]

        frag_df.to_csv(os.path.join(raw_folder, "frag.tsv"), sep="\t", index=False)
        psm_df.to_csv(os.path.join(raw_folder, "psm.tsv"), sep="\t", index=False)

    output = outputtransform.SearchPlanOutput(config, temp_folder)
    _ = output.build_precursor_table(raw_folders, save=True)
    _ = output.build_stat_df(raw_folders, save=True)
    _ = output.build_lfq_tables(raw_folders, save=True)

    # validate psm_df output
    psm_df = pd.read_csv(
        os.path.join(temp_folder, f"{output.PRECURSOR_OUTPUT}.tsv"), sep="\t"
    )
    assert all(
        [
            col in psm_df.columns
            for col in [
                "pg",
                "precursor_idx",
                "decoy",
                "mz_library",
                "charge",
                "proteins",
                "genes",
                "proba",
                "qval",
                "run",
            ]
        ]
    )
    assert psm_df["run"].nunique() == 3

    # validate stat_df output
    stat_df = pd.read_csv(
        os.path.join(temp_folder, f"{output.STAT_OUTPUT}.tsv"), sep="\t"
    )
    assert len(stat_df) == 3
    assert all([col in stat_df.columns for col in ["run", "precursors", "proteins"]])

    # validate protein_df output
    protein_df = pd.read_csv(os.path.join(temp_folder, f"pg.matrix.tsv"), sep="\t")
    assert all([col in protein_df.columns for col in ["run_0", "run_1", "run_2"]])

    for i in run_columns:
        for j in run_columns:
            if i == j:
                continue
            assert np.corrcoef(protein_df[i], protein_df[j])[0, 0] > 0.5

    import shutil

    shutil.rmtree(temp_folder)
