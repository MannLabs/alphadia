import numpy as np
import pandas as pd

from alphadia.constants.keys import CalibCols
from alphadia.fdr import fdr, interference

WINNER_MZ = 500.0
SHARED_FRAGMENT_MZ = 700.0


def _psms(**overrides) -> pd.DataFrame:
    row = {
        "precursor_idx": 1,
        "rank": 0,
        "charge": 2,
        interference.SEQUENCE_ID_COLUMN: 1,
        "fwhm_rt": 2.0,
        CalibCols.RT_OBSERVED: 100.0,
        CalibCols.MZ_OBSERVED: WINNER_MZ,
        "proba": 0.5,
    }
    return pd.DataFrame([row | overrides])


def _fragments(precursor_idx: int, mzs: list[float], number: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "precursor_idx": precursor_idx,
            "rank": 0,
            "number": number,
            "mz_library": mzs,
        }
    )


def _winner() -> pd.DataFrame:
    return _psms(precursor_idx=0, **{interference.SEQUENCE_ID_COLUMN: 0, "proba": 0.01})


def _run(candidate: pd.DataFrame, fragments: pd.DataFrame) -> bool:
    all_fragments = pd.concat([_fragments(0, [SHARED_FRAGMENT_MZ, 800.0]), fragments])
    return bool(
        interference.find_interfered_targets(candidate, _winner(), all_fragments)[0]
    )


def test_candidate_sharing_a_fragment_and_the_precursor_mass_is_removed():
    # Given: a candidate at the winner's m/z and RT that shares one ion of number >= 3
    candidate = _psms()

    # When / Then: it is flagged
    assert _run(candidate, _fragments(1, [SHARED_FRAGMENT_MZ, 900.0]))


def test_candidate_one_isotope_off_is_removed():
    # Given: a candidate whose M+1 sits on the winner's monoisotopic m/z
    candidate = _psms(
        **{CalibCols.MZ_OBSERVED: WINNER_MZ - interference.ISOTOPE_SPACING / 2}
    )

    # When / Then: it is flagged
    assert _run(candidate, _fragments(1, [SHARED_FRAGMENT_MZ]))


def test_candidate_is_kept_without_a_shared_fragment():
    # Given: same mass and RT, no shared fragment
    assert not _run(_psms(), _fragments(1, [900.0]))


def test_candidate_is_kept_if_only_a_low_number_ion_is_shared():
    # Given: the only shared ion is a b2/y2-like one
    assert not _run(_psms(), _fragments(1, [SHARED_FRAGMENT_MZ], number=2))


def test_candidate_is_kept_at_another_precursor_mass():
    # Given: a shared fragment but a precursor 20 ppm away
    candidate = _psms(**{CalibCols.MZ_OBSERVED: WINNER_MZ * (1 + 20e-6)})
    assert not _run(candidate, _fragments(1, [SHARED_FRAGMENT_MZ]))


def test_candidate_is_kept_outside_the_peak_width():
    # Given: everything shared, but 5 s apart with fwhm 2 s
    candidate = _psms(**{CalibCols.RT_OBSERVED: 105.0})
    assert not _run(candidate, _fragments(1, [SHARED_FRAGMENT_MZ]))


def test_candidate_of_the_same_sequence_is_kept():
    # Given: another charge state or modified form of the winner's peptide
    candidate = _psms(**{interference.SEQUENCE_ID_COLUMN: 0})
    assert not _run(candidate, _fragments(1, [SHARED_FRAGMENT_MZ]))


def test_candidate_better_than_the_winner_is_kept():
    # Given: a candidate more target-like (lower P(decoy)) than the winner
    candidate = _psms(proba=0.001)
    assert not _run(candidate, _fragments(1, [SHARED_FRAGMENT_MZ]))


def test_candidates_without_fragments_are_kept():
    # Given: a candidate with no quantified fragments
    candidates = pd.concat([_psms(), _psms(precursor_idx=2)], ignore_index=True)
    fragments = pd.concat(
        [_fragments(0, [SHARED_FRAGMENT_MZ]), _fragments(1, [SHARED_FRAGMENT_MZ])]
    )

    # When
    flagged = interference.find_interfered_targets(candidates, _winner(), fragments)

    # Then: only the candidate with the shared fragment is flagged
    np.testing.assert_array_equal(flagged, [True, False])


class _RowOrderClassifier:
    """Rank the PSMs by row order (targets first), counting the fits."""

    def __init__(self):
        self.n_fits = 0
        self.fitted = False

    def fit(self, x, y):
        self.n_fits += 1
        self.fitted = True

    def reset(self):
        self.fitted = False

    def predict_proba(self, x):
        proba = np.linspace(0.0, 1.0, len(x))
        return np.stack([1 - proba, proba], axis=1)


def test_perform_fdr_removes_interfered_candidates_and_refits_in_the_final_round():
    # Given: final-round PSMs where target 5 and the first decoy share mass, RT and a fragment with the better target 0
    n = 50
    target_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(n),
            "rank": 0,
            "decoy": 0,
            "feature": np.linspace(0.0, 1.0, n),
            "charge": 2,
            interference.SEQUENCE_ID_COLUMN: np.arange(n),
            "fwhm_rt": 2.0,
            CalibCols.RT_OBSERVED: 100.0 + 10.0 * np.arange(n),
            CalibCols.MZ_OBSERVED: 500.0 + np.arange(n),
        }
    )
    target_df.loc[5, [CalibCols.RT_OBSERVED, CalibCols.MZ_OBSERVED]] = [100.0, 500.0]
    decoy_df = target_df.assign(
        precursor_idx=np.arange(n, 2 * n),
        decoy=1,
        **{interference.SEQUENCE_ID_COLUMN: np.arange(n, 2 * n)},
    )
    for df in (target_df, decoy_df):
        df[fdr.ENSEMBLE_TRIGGER_FEATURE] = df["feature"]
        df["_candidate_idx"] = df["precursor_idx"]
    fragment_mz = 1000.0 + np.arange(2 * n, dtype=float)
    fragment_mz[5] = fragment_mz[0]
    fragment_mz[n] = fragment_mz[0]

    def provider(psm_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "precursor_idx": psm_df["precursor_idx"].to_numpy(),
                "rank": 0,
                "number": 5,
                "mz_library": fragment_mz[psm_df["precursor_idx"].to_numpy()],
            }
        )

    classifier = _RowOrderClassifier()

    # When: perform_fdr runs with a fragment source
    psm_df = fdr.perform_fdr(
        classifier,
        ["feature", fdr.ENSEMBLE_TRIGGER_FEATURE],
        target_df,
        decoy_df,
        fragment_provider=provider,
    )

    # Then: target 5 and that decoy are gone, everything else is kept, and the final round was fitted again
    targets = set(psm_df.loc[psm_df["decoy"] == 0, "precursor_idx"])
    decoys = set(psm_df.loc[psm_df["decoy"] == 1, "precursor_idx"])
    assert targets == set(range(n)) - {5}
    assert decoys == set(range(n, 2 * n)) - {n}
    assert classifier.n_fits == 2
