"""Unit test for the peptidecentric module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alphadia.fdr.classifiers import LightGBMClassifier
from alphadia.fdr.cross_fitting import CrossFittedTrainer
from alphadia.fdr.prefilter import CascadePrefilter
from alphadia.workflow.peptidecentric.optimization_handler import OptimizationHandler
from alphadia.workflow.peptidecentric.peptidecentric import (
    PeptideCentricWorkflow,
    _get_classifier_base,
    _get_prefilter,
    _get_trainer,
)


@pytest.fixture
def mock_config():
    return {
        "general": {"save_figures": True, "reuse_calibration": False},
        "output_directory": "",
        "calibration": {"min_correlation": 0.55, "max_fragments": 3},
    }


def test_filters_precursors_and_fragments_correctly(mock_config):
    """Test that the filter_dfs method filters precursors and fragments correctly."""
    precursor_df = pd.DataFrame(
        {"qval": [0.005, 0.005, 0.011], "decoy": [0, 1, 0], "precursor_idx": [1, 2, 3]}
    )
    fragments_df = pd.DataFrame(
        {
            "precursor_idx": [1, 1, 1, 2, 1, 1],
            "mass_error": [1, 3, -2, 5, -201, 1],
            "correlation": [0.7, 0.5, 0.8, 0.6, 0.9, 0.95],
        }
    )
    instance = PeptideCentricWorkflow("test_instance", mock_config)
    instance.reporter = MagicMock()
    with patch(
        "alphadia.workflow.peptidecentric.optimization_handler.OptimizationLock"
    ):
        instance._optimization_handler = OptimizationHandler(
            mock_config,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    # when
    filtered_precursors, filtered_fragments = (
        instance._optimization_handler._filter_dfs(precursor_df, fragments_df)
    )

    pd.testing.assert_frame_equal(
        filtered_precursors,
        pd.DataFrame(
            {
                "qval": [0.005],
                "decoy": [0],
                "precursor_idx": [1],
            }
        ),
    )

    pd.testing.assert_frame_equal(
        filtered_fragments.reset_index(drop=True),
        pd.DataFrame(
            {
                "precursor_idx": [1, 1, 1],
                "mass_error": [1, -2, 1],
                "correlation": [0.95, 0.8, 0.7],
            }
        ),
        check_like=True,
    )


def _classifier_config(name: str) -> dict:
    return {
        "general": {"thread_count": 2},
        "fdr": {
            "classifier": name,
            "enable_nn_hyperparameter_tuning": False,
            "lightgbm": {"n_estimators": 10, "final_n_estimators": 10},
        },
    }


def test_get_classifier_reads_the_lightgbm_configuration():
    classifier = _get_classifier_base(_classifier_config("lightgbm"), random_state=1)

    assert isinstance(classifier, LightGBMClassifier)
    assert classifier.n_estimators == 10
    assert classifier.num_threads == 2


def test_get_classifier_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown FDR classifier"):
        _get_classifier_base(_classifier_config("forest"))


def _prefilter_config(feature_subset: list[str], enabled: bool = True) -> dict:
    return {
        "general": {"thread_count": 2},
        "fdr": {
            "lightgbm": {"n_estimators": 10, "final_n_estimators": 10},
            "prefilter": {
                "enabled": enabled,
                "q_value_threshold": 0.3,
                "n_folds": 3,
                "n_estimators": 20,
                "final_n_estimators": 40,
                "max_train_psms": 1000,
                "feature_subset": feature_subset,
            },
        },
    }


def test_get_prefilter_is_none_when_disabled():
    assert _get_prefilter(_prefilter_config(["a"], enabled=False), ["a", "b"]) is None


def test_get_prefilter_reads_the_configuration_and_keeps_the_backend_order():
    prefilter = _get_prefilter(_prefilter_config(["c", "a"]), ["a", "b", "c"])

    assert isinstance(prefilter, CascadePrefilter)
    assert prefilter.feature_columns == ["a", "c"]
    assert prefilter.q_value_threshold == 0.3
    assert prefilter.n_folds == 3
    assert prefilter.max_train_psms == 1000
    assert prefilter._classifier.n_estimators == 20
    assert prefilter._classifier.final_n_estimators == 40
    assert prefilter._classifier.num_threads == 2


def test_get_prefilter_uses_every_feature_for_an_empty_subset():
    prefilter = _get_prefilter(_prefilter_config([]), ["a", "b"])

    assert prefilter.feature_columns == ["a", "b"]


def test_get_prefilter_rejects_an_unknown_feature():
    with pytest.raises(ValueError, match="does not provide"):
        _get_prefilter(_prefilter_config(["a", "typo"]), ["a", "b"])


def _cross_fitting_config(enabled: bool = True) -> dict:
    return {
        "fdr": {
            "cross_fitting": {
                "enabled": enabled,
                "n_folds": 4,
                "train_fdr": 0.02,
                "n_refits": 3,
            }
        }
    }


def test_get_trainer_is_none_when_disabled():
    assert _get_trainer(_cross_fitting_config(enabled=False)) is None


def test_get_trainer_reads_the_configuration():
    trainer = _get_trainer(_cross_fitting_config(), random_state=1)

    assert isinstance(trainer, CrossFittedTrainer)
    assert trainer.n_folds == 4
    assert trainer.train_fdr == 0.02
    assert trainer.n_refits == 3
