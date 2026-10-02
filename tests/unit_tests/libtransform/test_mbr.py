import numpy as np
import pandas as pd
import pytest
from alphabase.spectral_library.base import SpecLibBase, hash_precursor_df
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.libtransform.mbr import (
    IndexBuilder,
    MbrLibraryBuilder,
    bound_by_library_pg_qval,
    leave_one_out_rt,
)


class TestIndexBuilder:
    """Tests for IndexBuilder class."""

    def test_fallback_and_specific_lookup(self):
        """Test fallback lookup with partial specific overrides."""
        # given
        target_keys = np.array([100, 200, 300, 400])
        target_fallback_keys = np.array([0, 0, 1, 1])
        fallback_lookup_keys = np.array([0, 1])
        specific_lookup_keys = np.array([200, 400])

        fallback_values = np.array([10.0, 20.0])
        specific_values = np.array([100.0, 200.0])

        # when
        index_builder = IndexBuilder(
            target_keys,
            target_fallback_keys,
            fallback_lookup_keys,
            specific_lookup_keys,
        )
        result = index_builder.apply(fallback_values, specific_values)

        # then - targets 100, 300 get fallback; 200, 400 get specific
        np.testing.assert_array_equal(result, [10.0, 100.0, 20.0, 200.0])

    def test_empty_specific_keys(self):
        """Test with no specific keys (all fallback)."""
        # given
        target_keys = np.array([100, 200, 300])
        target_fallback_keys = np.array([0, 1, 2])
        fallback_lookup_keys = np.array([0, 1, 2])

        fallback_values = np.array([10.0, 20.0, 30.0])
        specific_values = np.array([])

        # when
        index_builder = IndexBuilder(
            target_keys, target_fallback_keys, fallback_lookup_keys, np.array([])
        )
        result = index_builder.apply(fallback_values, specific_values)

        # then
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_unsorted_lookup_keys(self):
        """Test with unsorted fallback lookup keys."""
        # given
        target_keys = np.array([100, 200, 300])
        target_fallback_keys = np.array([2, 0, 1])
        fallback_lookup_keys = np.array([1, 2, 0])

        fallback_values = np.array([10.0, 20.0, 30.0])
        specific_values = np.array([])

        # when
        index_builder = IndexBuilder(
            target_keys, target_fallback_keys, fallback_lookup_keys, np.array([])
        )
        result = index_builder.apply(fallback_values, specific_values)

        # then - target_fallback_keys [2,0,1] map to indices [1,2,0] in fallback_lookup_keys
        np.testing.assert_array_equal(result, [20.0, 30.0, 10.0])

    def test_numeric_and_string_values(self):
        """Test applying indices to both numeric and string arrays."""
        # given
        target_keys = np.array([100, 200, 300, 400])
        target_fallback_keys = np.array([0, 1, 2, 0])
        fallback_lookup_keys = np.array([0, 1, 2])
        specific_lookup_keys = np.array([200, 300])

        # when
        index_builder = IndexBuilder(
            target_keys,
            target_fallback_keys,
            fallback_lookup_keys,
            specific_lookup_keys,
        )

        # then - numeric
        result_num = index_builder.apply(
            np.array([10.0, 20.0, 30.0]),
            np.array([100.0, 200.0]),
        )
        np.testing.assert_array_equal(result_num, [10.0, 100.0, 200.0, 10.0])

        # then - string
        result_str = index_builder.apply(
            np.array(["A", "B", "C"]),
            np.array(["X", "Y"]),
        )
        np.testing.assert_array_equal(result_str, ["A", "X", "Y", "A"])

    def test_no_specific_matches(self):
        """Test when no specific matches exist (all fallback)."""
        # given
        target_keys = np.array([100, 200, 300, 400])
        target_fallback_keys = np.array([0, 1, 0, 1])
        fallback_lookup_keys = np.array([0, 1])
        specific_lookup_keys = np.array([999])

        fallback_values = np.array([10.0, 20.0])
        specific_values = np.array([100.0])

        # when
        index_builder = IndexBuilder(
            target_keys,
            target_fallback_keys,
            fallback_lookup_keys,
            specific_lookup_keys,
        )
        result = index_builder.apply(fallback_values, specific_values)

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
                "rt_calibrated": [12.0, 12.0, 22.0, 32.0],
                "pg_qval": [0.001, 0.001, 0.002, 0.002],
                "run": ["run_a", "run_a", "run_b", "run_a"],
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
                "rt_calibrated": [12.0, 22.0],
                "pg_qval": [0.001, 0.001],
                "run": ["run_a", "run_a"],
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


def test_mbr_library_builder_records_target_rt_by_run():
    # given
    lib = SpecLibBase()
    lib._precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDER", "PEPTIDEK"],
            "charge": [2, 2],
            "mods": ["", ""],
            "mod_sites": ["", ""],
        }
    )
    lib._precursor_df["nAA"] = lib._precursor_df["sequence"].str.len()
    lib.calc_precursor_mz()
    lib.calc_fragment_mz_df()
    lib._precursor_df["elution_group_idx"] = [0, 1]
    lib._precursor_df["precursor_idx"] = [0, 1]
    lib._precursor_df["decoy"] = 0
    lib._precursor_df["channel"] = 0
    lib._precursor_df = hash_precursor_df(lib._precursor_df)
    hashes = lib.precursor_df["mod_seq_charge_hash"].values
    psm_df = pd.DataFrame(
        {
            "elution_group_idx": [0, 0, 0, 1],
            "decoy": [0, 0, 1, 0],
            "qval": [0.001, 0.002, 0.001, 0.5],
            "rt_observed": [10.0, 14.0, 99.0, 30.0],
            "rt_calibrated": [11.0, 13.0, 99.0, 31.0],
            "pg_qval": [0.001, 0.001, 0.001, 0.001],
            "run": ["run_a", "run_b", "run_a", "run_a"],
            "pg": ["PG_A", "PG_A", "PG_A", "PG_B"],
            "mod_seq_charge_hash": [hashes[0], hashes[0], -1, hashes[1]],
        }
    )

    # when
    builder = MbrLibraryBuilder(fdr=0.01, keep_decoys=False)
    builder(psm_df, lib)

    # then
    rt_by_run = builder.rt_by_run.sort_values("run").reset_index(drop=True)
    np.testing.assert_array_equal(rt_by_run["elution_group_idx"], [0, 0])
    np.testing.assert_array_equal(rt_by_run["run"], ["run_a", "run_b"])
    np.testing.assert_array_equal(rt_by_run["rt_observed"], [10.0, 14.0])
    np.testing.assert_array_equal(rt_by_run["rt_calibrated"], [11.0, 13.0])


def test_leave_one_out_rt():
    # given
    speclib = SpecLibFlat()
    speclib._precursor_df = pd.DataFrame(
        {
            "elution_group_idx": [0, 0, 1, 2, 3],
            "decoy": [0, 1, 0, 0, 0],
            "rt_library": [100.0, 100.0, 200.0, 300.0, 400.0],
        }
    )
    rt_by_run = pd.DataFrame(
        {
            "elution_group_idx": [0, 0, 0, 1, 2],
            "run": ["run_a", "run_b", "run_c", "run_a", "run_b"],
            "rt_observed": [10.0, 20.0, 40.0, 50.0, 60.0],
            "rt_calibrated": [11.0, 21.0, 41.0, 51.0, 61.0],
        }
    )

    # when
    result = leave_one_out_rt(speclib, rt_by_run, "run_a")

    # then: group 0 from runs b and c, group 1 seen in run_a only keeps its calibrated RT there,
    # group 2 from run_b, group 3 never seen keeps the library RT
    np.testing.assert_array_equal(
        result.precursor_df["rt_library"], [30.0, 30.0, 51.0, 60.0, 400.0]
    )
    np.testing.assert_array_equal(
        speclib.precursor_df["rt_library"], [100.0, 100.0, 200.0, 300.0, 400.0]
    )


class TestMbrProteinFdr:
    """Tests for the protein-level filtering and q-value bounds of the MBR library."""

    @pytest.fixture
    def base_library(self):
        lib = SpecLibBase()
        lib._precursor_df = pd.DataFrame(
            {
                "sequence": ["PEPTIDER", "PEPTIDEK"],
                "charge": [2, 2],
                "mods": ["", ""],
                "mod_sites": ["", ""],
            }
        )
        lib._precursor_df["nAA"] = lib._precursor_df["sequence"].str.len()
        lib.calc_precursor_mz()
        lib.calc_fragment_mz_df()
        lib._precursor_df["elution_group_idx"] = [0, 1]
        lib._precursor_df["precursor_idx"] = [0, 1]
        lib._precursor_df["decoy"] = 0
        lib._precursor_df["channel"] = 0
        lib._precursor_df = hash_precursor_df(lib._precursor_df)
        return lib

    def test_groups_above_the_protein_fdr_are_left_out(self, base_library):
        # given: both precursors pass, but the second one's protein group only at 5 %
        hashes = base_library.precursor_df["mod_seq_charge_hash"].values
        psm_df = pd.DataFrame(
            {
                "elution_group_idx": [0, 1],
                "decoy": [0, 0],
                "qval": [0.001, 0.001],
                "pg_qval": [0.002, 0.05],
                "rt_observed": [10.0, 20.0],
                "rt_calibrated": [11.0, 21.0],
                "run": ["run_a", "run_a"],
                "pg": ["PG_A", "PG_B"],
                "mod_seq_charge_hash": hashes,
            }
        )

        # when
        builder = MbrLibraryBuilder(fdr=0.01, keep_decoys=False)
        result = builder(psm_df, base_library)

        # then
        np.testing.assert_array_equal(result.precursor_df["elution_group_idx"], [0])
        assert builder.pg_qval.to_dict("list") == {"pg": ["PG_A"], "pg_qval": [0.002]}

    def test_bound_by_library_pg_qval(self):
        # given
        psm_df = pd.DataFrame(
            {"pg": ["PG_A", "PG_B", "PG_C"], "pg_qval": [0.001, 0.02, 0.003]}
        )
        library_pg_qval = pd.DataFrame(
            {"pg": ["PG_A", "PG_B"], "pg_qval": [0.008, 0.001]}
        )

        # when
        result = bound_by_library_pg_qval(psm_df, library_pg_qval)

        # then: raised to the library q-value, never lowered, unknown groups kept
        np.testing.assert_allclose(result["pg_qval"], [0.008, 0.02, 0.003])
