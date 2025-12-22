import numpy as np
import pandas as pd
import pytest
from alphabase.spectral_library.base import SpecLibBase, hash_precursor_df

from alphadia.libtransform.mbr import (
    MbrLibraryBuilder,
    _apply_lookup_indices,
    _compute_lookup_indices,
)


class TestComputeLookupIndices:
    """Tests for compute_lookup_indices function."""

    def test_fallback_and_specific_lookup(self):
        """Test fallback lookup with partial specific overrides."""
        # given
        target_keys = np.array([100, 200, 300, 400])
        target_fallback_keys = np.array([0, 0, 1, 1])
        fallback_lookup_keys = np.array([0, 1])
        specific_lookup_keys = np.array([200, 400])

        # when
        fallback_idx, specific_idx, specific_index_mask = _compute_lookup_indices(
            target_keys,
            target_fallback_keys,
            fallback_lookup_keys,
            specific_lookup_keys,
        )

        # then
        np.testing.assert_array_equal(fallback_idx, [0, 0, 1, 1])
        np.testing.assert_array_equal(specific_index_mask, [False, True, False, True])
        np.testing.assert_array_equal(specific_idx[[1, 3]], [0, 1])

    def test_empty_specific_keys(self):
        """Test with no specific keys (all fallback)."""
        # given
        target_keys = np.array([100, 200, 300])
        target_fallback_keys = np.array([0, 1, 2])
        fallback_lookup_keys = np.array([0, 1, 2])

        # when
        fallback_idx, _, specific_index_mask = _compute_lookup_indices(
            target_keys, target_fallback_keys, fallback_lookup_keys, np.array([])
        )

        # then
        np.testing.assert_array_equal(specific_index_mask, [False, False, False])
        np.testing.assert_array_equal(fallback_idx, [0, 1, 2])

    def test_unsorted_lookup_keys(self):
        """Test with unsorted fallback lookup keys."""
        # given
        target_keys = np.array([100, 200, 300])
        target_fallback_keys = np.array([2, 0, 1])
        fallback_lookup_keys = np.array([1, 2, 0])

        # when
        fallback_idx, _, _ = _compute_lookup_indices(
            target_keys, target_fallback_keys, fallback_lookup_keys, np.array([])
        )

        # then
        np.testing.assert_array_equal(fallback_idx, [1, 2, 0])


class TestApplyLookupIndices:
    """Tests for apply_lookup_indices function."""

    def test_numeric_and_string_values(self):
        """Test applying indices to both numeric and string arrays."""
        # given
        fallback_indices = np.array([0, 1, 2, 0])
        specific_indices = np.array([0, 0, 1, 0])
        specific_index_mask = np.array([False, True, True, False])

        # when / then - numeric
        result_num = _apply_lookup_indices(
            np.array([10.0, 20.0, 30.0]),
            np.array([100.0, 200.0]),
            fallback_indices,
            specific_indices,
            specific_index_mask,
        )
        np.testing.assert_array_equal(result_num, [10.0, 100.0, 200.0, 10.0])

        # when / then - string
        result_str = _apply_lookup_indices(
            np.array(["A", "B", "C"]),
            np.array(["X", "Y"]),
            fallback_indices,
            specific_indices,
            specific_index_mask,
        )
        np.testing.assert_array_equal(result_str, ["A", "X", "Y", "A"])

    def test_no_specific_matches(self):
        """Test when no specific matches exist (all fallback)."""
        # given
        fallback_values = np.array([10.0, 20.0])
        specific_values = np.array([100.0])
        fallback_indices = np.array([0, 1, 0, 1])
        specific_indices = np.array([0, 0, 0, 0])
        specific_index_mask = np.array([False, False, False, False])

        # when
        result = _apply_lookup_indices(
            fallback_values,
            specific_values,
            fallback_indices,
            specific_indices,
            specific_index_mask,
        )

        # then
        np.testing.assert_array_equal(result, [10.0, 20.0, 10.0, 20.0])


class TestMbrLibraryBuilder:
    """Tests for the MbrLibraryBuilder class."""

    @pytest.fixture
    def base_library(self):
        """Create a minimal base library with 3 elution groups."""
        lib = SpecLibBase()
        lib._precursor_df = pd.DataFrame(
            {
                "sequence": ["PEPTIDER", "PEPTIDEK", "PEPTIDEA"],
                "charge": [2, 2, 2],
                "mods": ["", "", ""],
                "mod_sites": ["", "", ""],
            }
        )
        lib._precursor_df["nAA"] = lib._precursor_df["sequence"].str.len()
        lib.calc_precursor_mz()
        lib.calc_fragment_mz_df()
        lib._precursor_df["elution_group_idx"] = np.arange(len(lib._precursor_df))
        lib._precursor_df["precursor_idx"] = np.arange(len(lib._precursor_df))
        lib._precursor_df["decoy"] = 0
        lib._precursor_df["channel"] = 0
        lib._precursor_df = hash_precursor_df(lib._precursor_df)
        return lib

    @pytest.fixture
    def psm_df(self, base_library):
        """Create PSM dataframe with mixed FDR and identification scenarios."""
        lib_hashes = base_library.precursor_df["mod_seq_charge_hash"].values
        return pd.DataFrame(
            {
                "elution_group_idx": [0, 0, 1, 2],
                "decoy": [0, 1, 0, 0],
                "qval": [0.001, 0.005, 0.002, 0.5],
                "rt_observed": [10.0, 11.0, 20.0, 30.0],
                "pg": ["PG_A", "PG_A", "PG_B", "PG_C"],
                "mod_seq_charge_hash": [
                    lib_hashes[0],
                    -1,
                    lib_hashes[1],
                    lib_hashes[2],
                ],
            }
        )

    def test_fdr_filtering_and_decoy_generation(self, base_library, psm_df):
        """Test FDR filtering excludes high qval groups, decoy generation adds decoys."""
        # when - with decoys
        builder = MbrLibraryBuilder(fdr=0.01, keep_decoys=True)
        result = builder(psm_df, base_library)

        # then - check exact elution groups included
        df = result.precursor_df.sort_values(
            ["elution_group_idx", "decoy"]
        ).reset_index(drop=True)
        np.testing.assert_array_equal(df["elution_group_idx"].values, [0, 0, 1, 1])
        np.testing.assert_array_equal(df["decoy"].values, [0, 1, 0, 1])

        # when - without decoys
        builder_no_decoy = MbrLibraryBuilder(fdr=0.01, keep_decoys=False)
        result_no_decoy = builder_no_decoy(psm_df, base_library)

        # then - only targets
        df_no_decoy = result_no_decoy.precursor_df.sort_values(
            "elution_group_idx"
        ).reset_index(drop=True)
        np.testing.assert_array_equal(df_no_decoy["elution_group_idx"].values, [0, 1])
        np.testing.assert_array_equal(df_no_decoy["decoy"].values, [0, 0])

    def test_rt_and_pg_assignment(self, base_library, psm_df):
        """Test RT and protein group assignment with fallback and specific values."""
        # when
        builder = MbrLibraryBuilder(fdr=0.01, keep_decoys=True)
        result = builder(psm_df, base_library)

        # then - group 0: target=10.0 (specific), decoy=10.5 (fallback median)
        group_0 = result.precursor_df[result.precursor_df["elution_group_idx"] == 0]
        target_0 = group_0[group_0["decoy"] == 0].iloc[0]
        decoy_0 = group_0[group_0["decoy"] == 1].iloc[0]

        assert target_0["rt"] == 10.0
        assert decoy_0["rt"] == 10.5
        assert target_0["genes"] == "PG_A"
        assert target_0["proteins"] == "PG_A"
        assert decoy_0["genes"] == "PG_A"
        assert decoy_0["proteins"] == "PG_A"

        # then - group 1: only target in PSM, both get RT=20.0
        group_1 = result.precursor_df[result.precursor_df["elution_group_idx"] == 1]
        target_1 = group_1[group_1["decoy"] == 0].iloc[0]
        decoy_1 = group_1[group_1["decoy"] == 1].iloc[0]

        assert target_1["rt"] == 20.0
        assert decoy_1["rt"] == 20.0
        assert target_1["genes"] == "PG_B"
        assert decoy_1["genes"] == "PG_B"

    def test_decoy_only_elution_groups(self, base_library):
        """Test elution groups identified only by decoys are handled correctly."""
        # given
        lib_hashes = base_library.precursor_df["mod_seq_charge_hash"].values
        psm_df = pd.DataFrame(
            {
                "elution_group_idx": [0, 1],
                "decoy": [0, 1],
                "qval": [0.001, 0.005],
                "rt_observed": [10.0, 20.0],
                "pg": ["PG_A", "PG_B"],
                "mod_seq_charge_hash": [lib_hashes[0], -1],
            }
        )

        # when - with keep_decoys=True: include decoy-only groups
        builder_keep = MbrLibraryBuilder(fdr=0.01, keep_decoys=True)
        result_keep = builder_keep(psm_df, base_library)

        # then
        df_keep = result_keep.precursor_df.sort_values(
            ["elution_group_idx", "decoy"]
        ).reset_index(drop=True)
        np.testing.assert_array_equal(df_keep["elution_group_idx"].values, [0, 0, 1, 1])
        np.testing.assert_array_equal(df_keep["decoy"].values, [0, 1, 0, 1])
        np.testing.assert_array_equal(df_keep["rt"].values, [10.0, 10.0, 20.0, 20.0])
        np.testing.assert_array_equal(
            df_keep["genes"].values, ["PG_A", "PG_A", "PG_B", "PG_B"]
        )

        # when - with keep_decoys=False: exclude decoy-only groups
        builder_exclude = MbrLibraryBuilder(fdr=0.01, keep_decoys=False)
        result_exclude = builder_exclude(psm_df, base_library)

        # then
        df_exclude = result_exclude.precursor_df.sort_values(
            "elution_group_idx"
        ).reset_index(drop=True)
        np.testing.assert_array_equal(df_exclude["elution_group_idx"].values, [0])
        np.testing.assert_array_equal(df_exclude["decoy"].values, [0])
        assert df_exclude["rt"].values[0] == 10.0
        assert df_exclude["genes"].values[0] == "PG_A"
