from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from alphabase.spectral_library.base import SpecLibBase, hash_precursor_df

from alphadia.constants.keys import MbrCols
from alphadia.libtransform.decoy import DecoyGenerator
from alphadia.libtransform.mbr import (
    IndexBuilder,
    MbrLibraryBuilder,
    PerRunMbrLibrary,
    _median_of_other_rows,
    get_run_identifications,
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
                "pg_qval": [0.001, 0.001, 0.001, 0.001],
                "run": ["run_a", "run_a", "run_b", "run_b"],
                "rt_observed": [10.0, 11.0, 20.0, 30.0],
                "rt_calibrated": [10.5, 11.5, 20.5, 30.5],
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

    def test_decoy_type_is_passed_to_decoy_generator(self, base_library, psm_df):
        """Test that the configured decoy_type is used for MBR decoy generation."""
        # when
        with patch(
            "alphadia.libtransform.mbr.DecoyGenerator", wraps=DecoyGenerator
        ) as mock_decoy_generator:
            result = MbrLibraryBuilder(
                fdr=0.01, keep_decoys=True, decoy_type="pseudo_reverse"
            )(psm_df, base_library)

        # then
        mock_decoy_generator.assert_called_once_with(decoy_type="pseudo_reverse")
        assert result.precursor_df["decoy"].sum() == 2

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
                "pg_qval": [0.001, 0.001],
                "run": ["run_a", "run_a"],
                "rt_observed": [10.0, 20.0],
                "rt_calibrated": [10.5, 20.5],
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

    def test_protein_groups_above_fdr_are_excluded(self, base_library, psm_df):
        """Test a group whose protein group only passed above the MBR library FDR stays out."""
        # given
        psm_df.loc[psm_df["elution_group_idx"] == 1, "pg_qval"] = 0.05

        # when
        result = MbrLibraryBuilder(fdr=0.01, keep_decoys=False)(psm_df, base_library)

        # then
        np.testing.assert_array_equal(
            result.precursor_df["elution_group_idx"].values, [0]
        )


class TestGetRunIdentifications:
    """Tests for get_run_identifications."""

    def test_keeps_the_target_precursors_passing_both_fdrs(self):
        # given
        psm_df = pd.DataFrame(
            {
                "run": ["run_a", "run_a", "run_a", "run_b"],
                "elution_group_idx": [0, 0, 1, 2],
                "mod_seq_charge_hash": [100, 100, 200, 300],
                "decoy": [0, 1, 0, 0],
                "qval": [0.001, 0.001, 0.001, 0.05],
                "pg_qval": [0.001, 0.001, 0.05, 0.001],
                "rt_observed": [10.0, 11.0, 20.0, 30.0],
                "rt_calibrated": [10.5, 11.5, 20.5, 30.5],
                "proba": [0.1, 0.2, 0.3, 0.4],
            }
        )

        # when
        result = get_run_identifications(psm_df, fdr=0.01)

        # then
        assert list(result.columns) == [
            "run",
            "elution_group_idx",
            "mod_seq_charge_hash",
            "rt_observed",
            "rt_calibrated",
        ]
        assert result["run"].tolist() == ["run_a"]
        assert result["mod_seq_charge_hash"].tolist() == [100]
        assert result["rt_observed"].tolist() == [10.0]
        assert result["rt_calibrated"].tolist() == [10.5]

    def test_without_calibrated_rt_column(self):
        # given
        psm_df = pd.DataFrame(
            {
                "run": ["run_a"],
                "elution_group_idx": [0],
                "mod_seq_charge_hash": [100],
                "decoy": [0],
                "qval": [0.001],
                "pg_qval": [0.001],
                "rt_observed": [10.0],
            }
        )

        # when
        result = get_run_identifications(psm_df, fdr=0.01)

        # then
        assert result["rt_calibrated"].isna().all()


def test_median_of_other_rows_matches_brute_force():
    # given: keys with one to five rows each
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 30, 100)
    values = rng.normal(size=100)

    # when
    result = _median_of_other_rows(keys, values)

    # then
    expected = [
        np.median(values[(keys == key) & (np.arange(100) != i)])
        if (keys == key).sum() > 1
        else np.nan
        for i, key in enumerate(keys)
    ]
    np.testing.assert_allclose(result, expected)


def test_median_of_other_rows_without_rows():
    assert len(_median_of_other_rows(np.array([], dtype=int), np.array([]))) == 0


class TestPerRunMbrLibrary:
    """Tests for the PerRunMbrLibrary class."""

    @pytest.fixture
    def speclib(self):
        """MBR library whose elution group 0 has three precursors and a decoy sharing the first one's hash."""
        speclib = SpecLibBase()
        speclib._precursor_df = pd.DataFrame(
            {
                "precursor_idx": np.arange(8),
                "elution_group_idx": [0, 0, 0, 0, 1, 1, 2, 3],
                "mod_seq_charge_hash": [100, 101, 100, 102, 200, 201, 300, 400],
                "decoy": [0, 0, 1, 0, 0, 0, 0, 0],
                "rt_library": [1.0, 1.1, 1.0, 1.2, 2.0, 2.1, 3.0, 4.0],
            }
        )
        return speclib

    @pytest.fixture
    def run_identifications_df(self):
        return pd.DataFrame(
            {
                "run": ["run_a", "run_a", "run_a", "run_b", "run_b", "run_b", "run_c"],
                "elution_group_idx": [0, 1, 2, 0, 0, 3, 0],
                "mod_seq_charge_hash": [100, 200, 300, 100, 101, 400, 100],
                "rt_observed": [10.0, 18.0, 30.0, 12.0, 13.0, 40.0, 16.0],
                "rt_calibrated": [10.5, 19.0, np.nan, 12.5, 13.5, 40.5, 16.5],
            }
        )

    def test_rt_from_other_runs(self, speclib, run_identifications_df):
        """Test the library RT of the elution groups the run identified comes from the other runs."""
        # when
        result = PerRunMbrLibrary(run_identifications_df).for_run(speclib, "run_a")

        # then
        precursor_df = result.precursor_df.set_index("precursor_idx")
        # 0, 2: median over run_b and run_c of the precursor; the decoy shares the target's hash
        # 1: identified in run_b only, not in run_a
        # 3: identified in no run, median over the other runs' medians of the elution group
        # 4: identified in run_a only, calibrated RT of run_a
        # 5: identified in no run, calibrated RT of run_a's elution group
        # 6: identified in run_a only, without calibrated RT: not searched
        # 7: elution group not identified in run_a, MBR library RT kept
        assert precursor_df.index.tolist() == [0, 1, 2, 3, 4, 5, 7]
        np.testing.assert_allclose(
            precursor_df["rt_library"].values,
            [14.0, 13.0, 14.0, 14.25, 19.0, 19.0, 4.0],
        )
        np.testing.assert_array_equal(
            precursor_df[MbrCols.IDENTIFIED_IN_RUN].values, [1, 1, 1, 1, 1, 1, 0]
        )

    def test_rt_from_other_runs_is_specific_to_each_run(
        self, speclib, run_identifications_df
    ):
        """Test another run gets the RT without its own identifications, from the same adapter."""
        # when
        per_run_mbr_library = PerRunMbrLibrary(run_identifications_df)
        per_run_mbr_library.for_run(speclib, "run_a")
        result = per_run_mbr_library.for_run(speclib, "run_b")

        # then
        precursor_df = result.precursor_df.set_index("precursor_idx")
        np.testing.assert_allclose(
            precursor_df["rt_library"].values,
            [13.0, 13.0, 13.0, 13.0, 2.0, 2.1, 3.0, 40.5],
        )
        np.testing.assert_array_equal(
            precursor_df[MbrCols.IDENTIFIED_IN_RUN].values, [1, 1, 1, 1, 0, 0, 0, 1]
        )

    def test_run_without_identifications(self, speclib, run_identifications_df):
        """Test a run that identified nothing keeps the MBR library RT."""
        # when
        result = PerRunMbrLibrary(run_identifications_df).for_run(speclib, "run_d")

        # then
        np.testing.assert_array_equal(
            result.precursor_df["rt_library"].values,
            speclib.precursor_df["rt_library"].values,
        )
        assert (result.precursor_df[MbrCols.IDENTIFIED_IN_RUN] == 0).all()

    def test_input_library_is_not_modified(self, speclib, run_identifications_df):
        # given
        precursor_df_before = speclib.precursor_df.copy()

        # when
        PerRunMbrLibrary(run_identifications_df).for_run(speclib, "run_a")

        # then
        pd.testing.assert_frame_equal(speclib.precursor_df, precursor_df_before)
