from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytest.importorskip("peptdeep_kontext")

from peptdeep_kontext.core.model_manager import ModelManagerConfig  # noqa: E402
from peptdeep_kontext.core.training import TrainingConfig  # noqa: E402

from alphadia.exceptions import GenericUserError  # noqa: E402
from alphadia.transferlearning import kontext  # noqa: E402

KONTEXT_CONFIG = {
    "adaptation": None,
    "model_dir": "/models",
    "model_version": "v2",
    "config_path": None,
    "tto_epochs": 10,
    "tto_min_steps": 300,
    "tto_batch_size": 1000,
    "tto_lr": 0.1,
    "transfer_learning_epochs": 50,
    "transfer_learning_batch_size": 2000,
    "transfer_learning_lr": 0.0001,
    "transfer_learning_model_trainable": "heads",
    "transfer_learning_val_fraction": 0.1,
    "transfer_learning_keep_best_epoch": True,
    "prediction_batch_size": 1000,
}


def test_resolve_model_dir_explicit():
    """An explicit model_dir is returned without downloading."""
    with patch(
        "peptdeep_kontext.utils.cfg.download_pretrained_models"
    ) as mock_download:
        assert kontext.resolve_model_dir(KONTEXT_CONFIG) == "/models"

    mock_download.assert_not_called()


def test_resolve_model_dir_downloads_requested_version():
    """Without model_dir, the requested version is downloaded to the cache."""
    with patch(
        "peptdeep_kontext.utils.cfg.download_pretrained_models",
        return_value="/cache/models_v3",
    ) as mock_download:
        result = kontext.resolve_model_dir(
            KONTEXT_CONFIG | {"model_dir": None, "model_version": "v3"}
        )

    assert result == "/cache/models_v3"
    mock_download.assert_called_once_with(
        target_dir=kontext._DEFAULT_MODEL_CACHE, version="v3"
    )


def test_load_config_file_without_path():
    assert kontext.load_config_file(None) == {}


def test_load_config_file_yaml(tmp_path):
    config_path = tmp_path / "pdk.yaml"
    config_path.write_text(yaml.dump({"tto_config": {"epochs": 3}}))

    assert kontext.load_config_file(str(config_path)) == {"tto_config": {"epochs": 3}}


def test_load_config_file_python(tmp_path):
    config_path = tmp_path / "pdk.py"
    config_path.write_text("pipeline_config = dict(tto_config=dict(epochs=3))\n")

    assert kontext.load_config_file(str(config_path)) == {"tto_config": {"epochs": 3}}


def test_build_config_precedence():
    """The config file overrides alphaDIA settings, fixed settings override both."""
    config = kontext.build_config(
        TrainingConfig,
        alphadia_settings={"epochs": 10, "lr": 0.1, "batch_size": 1000},
        file_settings={
            "epochs": 3,
            "context_regularization": 0.01,
            "dataset_config": {"overfit": True, "indicator_columns": ["raw_name"]},
        },
        fixed_settings={"lr": 0.5, "dataset_config": {"indicator_columns": None}},
    )

    assert config.epochs == 3  # file over alphaDIA
    assert config.batch_size == 1000  # alphaDIA over pdk default
    assert config.context_regularization == 0.01  # file only
    assert config.lr == 0.5  # fixed over alphaDIA
    assert config.dataset_config.overfit is True  # nested file setting kept
    assert config.dataset_config.indicator_columns is None  # fixed over file


def test_build_config_raises_on_unknown_setting():
    with pytest.raises(GenericUserError, match="epochz"):
        kontext.build_config(ModelManagerConfig, {}, {"epochz": 3}, {})


def test_get_torch_device_maps_gpu_to_cuda():
    with patch("alphadia.transferlearning.kontext.utils.get_torch_device") as mock:
        mock.return_value = "gpu"
        assert kontext.get_torch_device(True) == "cuda"
        mock.return_value = "cpu"
        assert kontext.get_torch_device(False) == "cpu"


@pytest.mark.parametrize("method", ["tto", "hybrid"])
@patch("peptdeep_kontext.utils.cfg.get_pretrained_model_dict_from_dir")
@patch("peptdeep_kontext.core.training.TrainingManager")
def test_run_adaptation(mock_training_manager, mock_get_model, method, tmp_path):
    """TTO learns the context with a frozen ContextModel; hybrid then trains ContextDownstream with it."""
    mock_get_model.side_effect = lambda path: {"pretrained_model_path": path}

    kontext.run_adaptation(
        method=method,
        transfer_library_path="/transfer.hdf",
        output_dir=str(tmp_path),
        kontext_config=KONTEXT_CONFIG,
        charged_frag_types=["b_z1", "y_z1"],
        use_gpu=False,
    )

    configs = [call.args[0] for call in mock_training_manager.call_args_list]
    run_dirs = [
        call.kwargs["save_dir"]
        for call in mock_training_manager.return_value.run.call_args_list
    ]
    assert len(configs) == (1 if method == "tto" else 2)

    tto_config = configs[0]
    assert tto_config.model_class == "ContextModel"
    assert tto_config.model_trainable == "none"
    assert tto_config.context_mode == "learned"
    assert tto_config.min_steps == 300
    assert tto_config.lib_loader_config.root_dirs == ["/transfer.hdf"]
    assert tto_config.lib_loader_config.frag_types == ["b_z1", "y_z1"]
    assert tto_config.dataset_config.indicator_columns is None
    assert tto_config.device == "cpu"
    assert tto_config.pretrained_model.pretrained_model_path == str(
        Path("/models/ContextModel")
    )
    assert run_dirs[0] == str(tmp_path)
    assert (tmp_path / "tto_config.yaml").is_file()

    if method == "hybrid":
        transfer_learning_config = configs[1]
        assert transfer_learning_config.model_class == "ContextDownstream"
        assert transfer_learning_config.model_trainable == "heads"
        assert transfer_learning_config.context_mode == "precomputed"
        assert transfer_learning_config.context_source_path == kontext.get_context_path(
            tmp_path
        )
        assert transfer_learning_config.keep_best_epoch is True
        assert run_dirs[1] == str(tmp_path / "transfer_learning")
        assert kontext.get_transfer_learned_model_path(tmp_path) == str(
            tmp_path / "transfer_learning" / "ContextDownstream"
        )
