import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from alphadia.constants.keys import FeatureTransform
from alphadia.fdr import fdr, riding
from alphadia.fdr.classifiers import BinaryClassifierLegacyNewBatching, Classifier


def test_keep_best():
    test_df = pd.DataFrame(
        {
            "precursor_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "channel": [0, 0, 1, 0, 1, 1, 0, 0, 1],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )

    best_df = fdr.keep_best(
        test_df, score_column="proba", group_columns=["precursor_idx"]
    )

    assert best_df.shape[0] == 3
    assert np.allclose(best_df["proba"].values, np.array([0.1, 0.4, 0.7]))

    best_df = fdr.keep_best(
        test_df, score_column="proba", group_columns=["channel", "precursor_idx"]
    )

    assert best_df.shape[0] == 6
    assert np.allclose(
        best_df["proba"].values, np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.9])
    )


def test_keep_best_2():
    test_df = pd.DataFrame(
        {
            "channel": [0, 0, 0, 4, 4, 4, 8, 8, 8],
            "elution_group_idx": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )

    result_df = fdr.keep_best(test_df, group_columns=["channel", "elution_group_idx"])
    pd.testing.assert_frame_equal(result_df, test_df)

    test_df = pd.DataFrame(
        {
            "channel": [0, 0, 0, 4, 4, 4, 8, 8, 8],
            "elution_group_idx": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )
    result_df = fdr.keep_best(test_df, group_columns=["channel", "elution_group_idx"])
    result_expected = pd.DataFrame(
        {
            "channel": [0, 0, 4, 4, 8, 8],
            "elution_group_idx": [0, 1, 0, 1, 0, 1],
            "proba": [0.1, 0.3, 0.4, 0.6, 0.1, 0.3],
        }
    )
    pd.testing.assert_frame_equal(result_df, result_expected)

    test_df = pd.DataFrame(
        {
            "channel": [0, 0, 0, 4, 4, 4, 8, 8, 8],
            "precursor_idx": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )
    result_df = fdr.keep_best(test_df, group_columns=["channel", "precursor_idx"])
    result_expected = pd.DataFrame(
        {
            "channel": [0, 0, 4, 4, 8, 8],
            "precursor_idx": [0, 1, 0, 1, 0, 1],
            "proba": [0.1, 0.3, 0.4, 0.6, 0.1, 0.3],
        }
    )
    pd.testing.assert_frame_equal(result_df, result_expected)


def test_fdr_to_q_values():
    test_fdr = np.array([0.2, 0.1, 0.05, 0.3, 0.26, 0.25, 0.5])

    test_q_values = fdr._fdr_to_q_values(test_fdr)

    assert np.allclose(
        test_q_values, np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 0.5])
    )


def test_get_q_values():
    test_df = pd.DataFrame(
        {
            "precursor_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "_decoy": [0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
        }
    )

    test_df = fdr.get_q_values(test_df, "proba", "_decoy")

    assert np.allclose(
        test_df["qval"].values,
        np.array([0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 1.0]),
    )


def test_get_q_values_all_tied_scores_are_conservative():
    # Given: an uninformative classifier assigning every PSM the same probability
    n_per_class = 500
    test_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(2 * n_per_class),
            "proba": np.full(2 * n_per_class, 0.1),
            "_decoy": np.repeat([0, 1], n_per_class),
        }
    )

    # When
    test_df = fdr.get_q_values(test_df, "proba", "_decoy")

    # Then: the tied block carries its decoy fraction, so nothing is accepted
    assert np.allclose(test_df["qval"].values, 1.0)
    assert (test_df["qval"].values < 0.01).sum() == 0


def test_get_q_values_tied_best_score_block_does_not_pass():
    # Given: 600 targets and 300 decoys tied at the best score, the rest at a worse one
    best_score_decoys, best_score_targets = 300, 600
    worst_score_decoys, worst_score_targets = 200, 100
    test_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(
                best_score_decoys
                + best_score_targets
                + worst_score_decoys
                + worst_score_targets
            ),
            "proba": np.repeat(
                [0.1, 0.9],
                [
                    best_score_decoys + best_score_targets,
                    worst_score_decoys + worst_score_targets,
                ],
            ),
            "_decoy": np.repeat(
                [1, 0, 1, 0],
                [
                    best_score_decoys,
                    best_score_targets,
                    worst_score_decoys,
                    worst_score_targets,
                ],
            ),
        }
    )

    # When
    test_df = fdr.get_q_values(test_df, "proba", "_decoy")

    # Then: no PSM passes 1% FDR and q-values stay monotone along the sorted output
    assert (test_df["qval"].values < 0.01).sum() == 0
    assert np.allclose(
        test_df["qval"].values[: best_score_decoys + best_score_targets],
        best_score_decoys / best_score_targets,
    )
    assert np.all(np.diff(test_df["qval"].values) >= 0)


def gen_data_np(
    n_features=10,
    n_samples=10000,
    max_mean=100,
    max_var=0.1,
):
    mean = np.random.random(n_features * 2) * max_mean
    var = np.random.random(n_features * 2) * max_var
    data = np.random.multivariate_normal(
        mean, np.eye(n_features * 2) * var, size=n_samples
    )
    return data.reshape(-1, n_features), np.tile([0, 1], n_samples)


def test_feed_forward():
    x, y = gen_data_np()

    classifier = BinaryClassifierLegacyNewBatching(
        batch_size=100,
    )

    classifier.fit(x, y)
    # assert classifier.metrics["test_accuracy"][-1] > 0.99
    # assert classifier.metrics["train_accuracy"][-1] > 0.99

    y_pred = classifier.predict(x)  # noqa: F841  # TODO fix this test
    # assert np.all(y_pred == y)

    y_proba = classifier.predict_proba(x)[:, 1]  # noqa: F841  # TODO fix this test
    # assert np.all(np.round(y_proba) == y)


def test_feed_forward_save():
    tempfolder = tempfile.gettempdir()
    x, y = gen_data_np()

    classifier = BinaryClassifierLegacyNewBatching(
        batch_size=100,
    )

    classifier.fit(x, y)

    torch.save(
        classifier.to_state_dict(),
        os.path.join(tempfolder, "test_feed_forward_save.pth"),
    )

    new_classifier = BinaryClassifierLegacyNewBatching()
    new_classifier.from_state_dict(
        torch.load(
            os.path.join(tempfolder, "test_feed_forward_save.pth"), weights_only=False
        )
    )

    y_pred = new_classifier.predict(x)  # noqa: F841  # TODO fix this test
    # assert np.all(y_pred == y)


def test_classifier_reset():
    # Given: a fitted classifier
    x, y = gen_data_np()
    classifier = BinaryClassifierLegacyNewBatching(batch_size=100)
    classifier.fit(x, y)
    first_weights = classifier.network.fc_layers[0].weight.detach().numpy().copy()

    # When: the test resets it
    classifier.reset()

    # Then: the classifier is unfitted, with no network and no metrics
    assert classifier.fitted is False
    assert classifier.network is None
    assert all(len(values) == 0 for values in classifier.metrics.values())

    # And: a new fit uses new random weights, not the first ones
    classifier.fit(x, y)
    second_weights = classifier.network.fc_layers[0].weight.detach().numpy()
    assert classifier.fitted is True
    assert not np.allclose(first_weights, second_weights)


class _CollapsingClassifier(Classifier):
    """Give a constant probability until the caller resets it `n_collapses` times."""

    def __init__(self, n_collapses: int):
        self._n_collapses = n_collapses
        self._fitted = False
        self.reset_count = 0

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, x, y):
        self._fitted = True

    def reset(self):
        self.reset_count += 1
        self._fitted = False

    def predict(self, x):
        return self.predict_proba(x)[:, 1]

    def predict_proba(self, x):
        if self.reset_count < self._n_collapses:
            return np.full((len(x), 2), 0.5)
        proba = np.linspace(0.0, 1.0, len(x))
        return np.stack([1 - proba, proba], axis=1)

    def to_state_dict(self):
        return {}

    def from_state_dict(self, state_dict):
        pass


def _gen_psm_df(n_samples: int = 200):
    feature = np.linspace(0.0, 1.0, n_samples)
    target_df = pd.DataFrame(
        {
            "precursor_idx": np.arange(n_samples),
            "decoy": 0,
            "feature": feature,
        }
    )
    decoy_df = target_df.assign(
        precursor_idx=np.arange(n_samples, 2 * n_samples), decoy=1
    )
    return pd.concat([target_df, decoy_df], ignore_index=True)


def test_perform_fdr_resets_collapsed_classifier():
    # Given: a classifier that collapses once, then separates targets from decoys
    classifier = _CollapsingClassifier(n_collapses=1)
    psm_df = _gen_psm_df()

    # When: perform_fdr runs
    psm_df = fdr.perform_fdr(
        classifier, ["feature"], psm_df, psm_df["decoy"].to_numpy() == 1
    )

    # Then: perform_fdr resets the classifier once and uses the new probabilities
    assert classifier.reset_count == 1
    assert psm_df["proba"].std() > 0.0


def test_perform_fdr_stops_after_max_reinits():
    # Given: a classifier that never recovers
    classifier = _CollapsingClassifier(n_collapses=1000)
    psm_df = _gen_psm_df()

    # When: perform_fdr runs
    psm_df = fdr.perform_fdr(
        classifier, ["feature"], psm_df, psm_df["decoy"].to_numpy() == 1
    )

    # Then: perform_fdr stops after the maximum number of retries
    assert classifier.reset_count == fdr._MAX_FDR_CLASSIFIER_REINITS
    assert psm_df["proba"].std() == 0.0


def test_fit_feature_transform_falls_back_to_raw_features_for_few_rows():
    # Given: a classifier with the quantile transform and fewer rows than the minimum
    classifier = BinaryClassifierLegacyNewBatching(
        feature_transform=FeatureTransform.QUANTILE
    )
    x, _ = gen_data_np(n_samples=100)

    # When: the transform is fitted
    transformer = classifier._fit_feature_transform(x)

    # Then: no transformer is fitted, so the raw features are used
    assert transformer is None


def test_classifier_raises_for_unknown_transform():
    # When/Then: constructing a classifier with an unknown transform fails explicitly
    with pytest.raises(ValueError, match="Unknown feature transform"):
        BinaryClassifierLegacyNewBatching(feature_transform="unknown")


def test_quantile_transform_state_dict_round_trip():
    # Given: a classifier fitted with the quantile transform
    x, y = gen_data_np(n_samples=2000)
    classifier = BinaryClassifierLegacyNewBatching(
        batch_size=100, feature_transform=FeatureTransform.QUANTILE
    )
    classifier.fit(x, y)
    assert classifier._quantile_transformer is not None

    # When: the state dict is round tripped through a new classifier
    new_classifier = BinaryClassifierLegacyNewBatching()
    new_classifier.from_state_dict(classifier.to_state_dict())

    # Then: the predictions are identical
    assert np.array_equal(new_classifier.predict_proba(x), classifier.predict_proba(x))


# one isolation window from 400 to 600 m/z, no mobility
_DIA_CYCLE = np.array([[[[400.0, 600.0]]]])


def _run_perform_fdr_with_mocked_competition(**fragment_kwargs):
    """Run perform_fdr with compete_for_fragments replaced by a pass-through mock.

    The competition itself is tested in alphadia-search-rs; this only checks which
    fragments perform_fdr hands to it.
    """
    psm_df = _gen_psm_df()
    with patch.object(
        fdr, "compete_for_fragments", side_effect=lambda psm_df, *_: psm_df
    ) as mock_compete:
        fdr.perform_fdr(
            _CollapsingClassifier(n_collapses=0),
            ["feature"],
            psm_df,
            psm_df["decoy"].to_numpy() == 1,
            dia_cycle=_DIA_CYCLE,
            **fragment_kwargs,
        )
    return mock_compete


def test_perform_fdr_quantifies_fragments_on_demand_for_the_competing_psms():
    # Given: a provider instead of fragments
    provider = MagicMock(name="provider")

    # When: perform_fdr runs
    mock_compete = _run_perform_fdr_with_mocked_competition(fragment_provider=provider)

    # Then: the provider is asked once, only for the PSMs below the heuristic q-value
    provider.assert_called_once()
    requested_psms = provider.call_args.args[0]
    assert len(requested_psms) > 0
    assert requested_psms["qval"].max() < 0.1
    # and its fragments go to the competition
    assert mock_compete.call_args.args[1] is provider.return_value


def test_perform_fdr_prefers_given_fragments_over_the_provider():
    # Given: fragments up front, and a provider
    df_fragments = pd.DataFrame()
    provider = MagicMock(name="provider")

    # When: perform_fdr runs with both
    mock_compete = _run_perform_fdr_with_mocked_competition(
        df_fragments=df_fragments, fragment_provider=provider
    )

    # Then: the given fragments go to the competition, and the provider is not asked
    provider.assert_not_called()
    assert mock_compete.call_args.args[1] is df_fragments


class _FirstFeatureClassifier:
    """Scores by the first feature; records how many rows it was fitted on."""

    fitted_rows = None

    def fit(self, X, y):  # noqa: N803
        self.fitted_rows = len(X)

    def predict_proba(self, X):  # noqa: N803
        p = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p, p])


def test_prescore_proba_fits_a_copy_on_a_sample_and_predicts_every_row(monkeypatch):
    # given: more rows than the pre-score may fit on
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 2))
    y = (X[:, 0] > 0).astype(float)
    classifier = _FirstFeatureClassifier()
    monkeypatch.setattr(fdr.riding, "PRESCORE_ROWS", 800)
    fitted = []
    original = fdr._fit_predict

    def spy(clf, X, y_train, idxs_train):  # noqa: N803
        fitted.append((clf, len(idxs_train), len(y_train)))
        return original(clf, X, y_train, idxs_train)

    monkeypatch.setattr(fdr, "_fit_predict", spy)

    # when
    proba = fdr._prescore_proba(classifier, X, y, random_state=1)

    # then
    assert proba.shape == (5000,)
    assert fitted[0][1:] == (800, 800)
    assert fitted[0][0] is not classifier
    assert classifier.fitted_rows is None


@pytest.mark.parametrize("with_provider", [True, False])
def test_perform_fdr_removes_riding_psms_before_the_fit(monkeypatch, with_provider):
    # given: a removal that drops the first ten target PSMs
    psm_df = _gen_psm_df()
    removed = set(range(10))
    calls = []

    def without_riding(psm_df, predicted_proba, *args):
        calls.append(len(predicted_proba))
        return np.flatnonzero(~psm_df["precursor_idx"].isin(removed).to_numpy())

    monkeypatch.setattr(fdr, "_without_riding", without_riding)
    classifier = _FirstFeatureClassifier()
    riding_kwargs = (
        {"riding_provider": MagicMock(name="provider"), "ms2_tolerance_ppm": 10.0}
        if with_provider
        else {}
    )

    # when
    with patch.object(fdr, "compete_for_fragments", side_effect=lambda df, *_: df):
        result = fdr.perform_fdr(
            classifier,
            ["feature"],
            psm_df,
            psm_df["decoy"].to_numpy() == 1,
            dia_cycle=_DIA_CYCLE,
            **riding_kwargs,
        )

    # then: with a provider, the pre-score sees every PSM and the classifier only the kept ones
    if with_provider:
        assert calls == [len(psm_df)]
        assert classifier.fitted_rows == int(0.8 * (len(psm_df) - len(removed)))
        assert not set(result["precursor_idx"]) & removed
    else:
        assert calls == []
        assert classifier.fitted_rows == int(0.8 * len(psm_df))


def test_perform_fdr_skips_riding_removal_with_ion_mobility(monkeypatch):
    # given: a DIA cycle with a mobility dimension
    psm_df = _gen_psm_df()
    mock_without_riding = MagicMock(name="without_riding")
    monkeypatch.setattr(fdr, "_without_riding", mock_without_riding)

    # when
    with patch.object(fdr, "compete_for_fragments", side_effect=lambda df, *_: df):
        fdr.perform_fdr(
            _FirstFeatureClassifier(),
            ["feature"],
            psm_df,
            psm_df["decoy"].to_numpy() == 1,
            dia_cycle=np.zeros((1, 1, 2, 2)),
            riding_provider=MagicMock(name="provider"),
            ms2_tolerance_ppm=10.0,
        )

    # then
    mock_without_riding.assert_not_called()


def _riding_psms() -> pd.DataFrame:
    """Two candidates per target precursor 0-39 (rank 0 better), one candidate per decoy precursor 40-49."""
    n_targets, n_decoys = 40, 10
    targets = pd.DataFrame(
        {
            "precursor_idx": np.repeat(np.arange(n_targets), 2),
            "rank": np.tile([0, 1], n_targets),
            "proba": np.repeat(np.linspace(0.01, 0.2, n_targets), 2)
            + np.tile([0.0, 0.5], n_targets),
            "_decoy": 0,
        }
    )
    decoys = pd.DataFrame(
        {
            "precursor_idx": np.arange(n_targets, n_targets + n_decoys),
            "rank": 0,
            "proba": np.linspace(0.9, 0.99, n_decoys),
            "_decoy": 1,
        }
    )
    psm_df = pd.concat([targets, decoys], ignore_index=True)
    return psm_df.assign(
        elution_group_idx=psm_df["precursor_idx"],
        _sequence_id=psm_df["precursor_idx"],
        fwhm_rt=2.0,
        rt_observed=100.0,
        mz_observed=500.0,
    )


def test_without_riding_removes_rows_above_the_riding_share(monkeypatch):
    # given: rows 2 and 5 ride (share above riding.MIN_RIDING), row 7 just below
    psm_df = _riding_psms()
    share = np.zeros(len(psm_df))
    share[[2, 5]] = riding.MIN_RIDING
    share[7] = riding.MIN_RIDING * 0.99
    received = {}

    def find_riding(rows, fragments, partner, cycle, ms2_tolerance_ppm):
        received["rows"], received["partner"] = rows, partner
        received["fragments"], received["tolerance"] = fragments, ms2_tolerance_ppm
        return share, np.zeros(len(rows))

    monkeypatch.setattr(riding, "find_riding", find_riding)
    provider = MagicMock(name="provider")

    # when
    keep = fdr._without_riding(
        psm_df,
        psm_df["proba"].to_numpy(),
        ["elution_group_idx"],
        provider,
        _DIA_CYCLE,
        7.0,
    )

    # then: the riding rows are dropped and every row is a removal candidate
    assert keep.tolist() == [i for i in range(len(psm_df)) if i not in (2, 5)]
    assert provider.call_args.args[0]["precursor_idx"].tolist() == psm_df[
        "precursor_idx"
    ].tolist()
    assert received["fragments"] is provider.return_value
    assert received["tolerance"] == 7.0
    # and the partners are the best candidates of the confident elution groups, never a second candidate
    partner = received["partner"]
    assert partner[psm_df["rank"].to_numpy() == 1].sum() == 0
    assert partner[(psm_df["rank"] == 0) & (psm_df["_decoy"] == 0)].all()


def test_without_riding_keeps_all_rows_without_riding(monkeypatch):
    # given: no row rides
    psm_df = _riding_psms()
    monkeypatch.setattr(
        riding,
        "find_riding",
        lambda rows, *_: (np.zeros(len(rows)), np.zeros(len(rows))),
    )

    # when
    keep = fdr._without_riding(
        psm_df,
        psm_df["proba"].to_numpy(),
        ["elution_group_idx"],
        MagicMock(),
        _DIA_CYCLE,
        7.0,
    )

    # then: nothing to remove
    assert keep is None


def test_without_riding_needs_its_columns():
    # given: PSMs without the sequence id
    psm_df = _riding_psms().drop(columns=riding.SEQUENCE_ID_COLUMN)
    provider = MagicMock(name="provider")

    # when
    keep = fdr._without_riding(
        psm_df,
        psm_df["proba"].to_numpy(),
        ["elution_group_idx"],
        provider,
        _DIA_CYCLE,
        7.0,
    )

    # then: no removal and no fragments quantified
    assert keep is None
    provider.assert_not_called()
