from unittest.mock import patch

from alphadia.transferlearning.context_extraction import (
    _DEFAULT_KONTEXT_CACHE,
    prepare_context_model_path,
)


def test_prepare_context_model_path_explicit():
    """When an explicit path is set, it is returned without triggering a download."""
    config = {"context_model_path": "/explicit/path/to/ContextModel"}
    with patch(
        "alphadia.transferlearning.context_extraction.download_pretrained_models"
    ) as mock_download:
        result = prepare_context_model_path(config)

    assert result == "/explicit/path/to/ContextModel"
    mock_download.assert_not_called()


def test_prepare_context_model_path_auto_download():
    """When path is null, weights are downloaded and the ContextModel subdir is returned."""
    config = {"context_model_version": "v8"}
    with patch(
        "alphadia.transferlearning.context_extraction.download_pretrained_models",
        return_value="/cache/peptdeep_kontext/models_v8",
    ) as mock_download:
        result = prepare_context_model_path(config)

    mock_download.assert_called_once_with(
        version="v8", target_dir=_DEFAULT_KONTEXT_CACHE
    )
    assert result == "/cache/peptdeep_kontext/models_v8/ContextModel"


def test_prepare_context_model_path_auto_download_default_version():
    """When context_model_version is missing, None is passed so the library uses its own default."""
    config = {}
    with patch(
        "alphadia.transferlearning.context_extraction.download_pretrained_models",
        return_value="/cache/peptdeep_kontext/models_v8",
    ) as mock_download:
        prepare_context_model_path(config)

    mock_download.assert_called_once_with(
        version=None, target_dir=_DEFAULT_KONTEXT_CACHE
    )


def test_prepare_context_model_path_custom_subdir():
    """Caller can override subdir for the ContextDownstream use case."""
    config = {}
    with patch(
        "alphadia.transferlearning.context_extraction.download_pretrained_models",
        return_value="/cache/peptdeep_kontext/models_v8",
    ):
        result = prepare_context_model_path(
            config, path_key="peptdeep_kontext_model_path", subdir="ContextDownstream"
        )

    assert result == "/cache/peptdeep_kontext/models_v8/ContextDownstream"


def test_prepare_context_model_path_explicit_overrides_subdir():
    """Explicit path is returned as-is, ignoring the subdir parameter."""
    config = {"peptdeep_kontext_model_path": "/user/path/ContextDownstream"}
    with patch(
        "alphadia.transferlearning.context_extraction.download_pretrained_models"
    ) as mock_download:
        result = prepare_context_model_path(
            config, path_key="peptdeep_kontext_model_path", subdir="ContextDownstream"
        )

    assert result == "/user/path/ContextDownstream"
    mock_download.assert_not_called()
