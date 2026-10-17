import pandas as pd
from alphabase.spectral_library.base import SpecLibBase

from alphadia.workflow.peptidecentric.library_init import save_library_precursors


def test_save_library_precursors_keeps_identity_columns(tmp_path):
    library = SpecLibBase()
    library._precursor_df = pd.DataFrame(
        {
            "precursor_idx": [7, 3],
            "elution_group_idx": [1, 1],
            "decoy": [0, 1],
            "sequence": ["PEPTIDEK", "PDPTIEEK"],
            "mods": ["", ""],
            "mod_sites": ["", ""],
            "charge": [2, 2],
            "proteins": ["P1", "P1"],
            "mz_library": [450.2, 451.2],
            "flat_frag_start_idx": [0, 10],
        }
    )
    path = tmp_path / "library_precursors.parquet"

    save_library_precursors(library, str(path))

    saved = pd.read_parquet(path)
    assert "flat_frag_start_idx" not in saved
    pd.testing.assert_frame_equal(
        saved, library.precursor_df.drop(columns="flat_frag_start_idx")
    )
