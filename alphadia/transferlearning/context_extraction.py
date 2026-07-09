"""Module for extracting context from raw files using peptdeep_kontext.

This is the alternative to the transfer learning module: instead of fine-tuning
a model on high-quality PSMs, it runs test-time optimization (TTO) to extract
instrument/sample-specific context embeddings.
"""

import logging
import os

from peptdeep_kontext.core.tto import TTOConfig, TTOManager
from peptdeep_kontext.datasets.dataset import DatasetConfig
from peptdeep_kontext.datasets.libloader import LibLoaderConfig
from peptdeep_kontext.utils.cfg import (
    download_pretrained_models,
    get_pretrained_model_config_from_dir,
)

from alphadia.utils import expand_path

logger = logging.getLogger(__name__)

_DEFAULT_KONTEXT_CACHE = expand_path("~/.cache/peptdeep_kontext")


def prepare_context_model_path(
    config: dict,
    path_key: str = "context_model_path",
    subdir: str = "ContextModel",
    version: str | None = None,
) -> str:
    """Return the peptdeep_kontext model path, downloading weights first if not set.

    Parameters
    ----------
    config : dict
        Config section containing the explicit model path under ``path_key``.
    path_key : str
        Key within ``config`` that holds the explicit model path.
        Defaults to ``"context_model_path"``.
    subdir : str
        Subdirectory within the downloaded model bundle to return when the
        explicit path is not set. Defaults to ``"ContextModel"``.
    version : str, optional
        Weights version tag used when auto-downloading, sourced by the caller
        from ``context_extraction.context_model_version``. If None, the
        downloader's own default version is used.
    """
    model_path = config.get(path_key)
    if model_path:
        return expand_path(model_path)

    logger.info(
        f"{path_key} not set — downloading peptdeep_kontext weights "
        f"(version={version or 'default'}) to {_DEFAULT_KONTEXT_CACHE}"
    )
    # Don't forward a None version: download_pretrained_models has its own
    # default, and passing None would override it and build a "models_None" URL.
    version_kwargs = {"version": version} if version is not None else {}
    base_dir = download_pretrained_models(
        target_dir=_DEFAULT_KONTEXT_CACHE, **version_kwargs
    )
    return os.path.join(base_dir, subdir)


class ContextExtractor:
    def __init__(
        self,
        annotated_speclib_path: str,
        charged_frag_types: list[str],
        pretrained_context_model_path: str,
        tto_epoch: int = 10,
        tto_batch_size: int = 1000,
        tto_lr: float = 0.1,
        tto_warmup_epochs: int = 5,
        context_indicator_columns: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the ContextExtractor.

        Parameters
        ----------
        annotated_speclib_path : str
            Path to the annotated spectral library (output of a first-pass search).
        charged_frag_types : list[str]
            Charged fragment types to use during TTO.
        pretrained_context_model_path : str
            Path to the pretrained peptdeep_kontext context model directory.
        tto_epoch : int, optional
            Number of epochs for test-time optimization, by default 10.
        tto_batch_size : int, optional
            Batch size for test-time optimization, by default 1000.
        tto_lr : float, optional
            Learning rate for test-time optimization, by default 0.1.
        tto_warmup_epochs : int, optional
            Number of warmup epochs for test-time optimization, by default 5.
        context_indicator_columns : list[str], optional
            Columns that define the context grouping (e.g. ``['raw_name']`` for
            per-raw-file context or ``['constant_context_indicator']`` for a
            single shared context). Defaults to ``['constant_context_indicator']``.
        verbose : bool, optional
            Whether to print verbose output during TTO, by default False.
        """
        if context_indicator_columns is None:
            context_indicator_columns = ["constant_context_indicator"]

        tto_config = TTOConfig(
            tto_epochs=tto_epoch,
            tto_batch_size=tto_batch_size,
            tto_lr=tto_lr,
            tto_warmup_epochs=tto_warmup_epochs,
            verbose=verbose,
            pretrained_context_model=get_pretrained_model_config_from_dir(
                pretrained_context_model_path
            ),
            tto_dataset_config=DatasetConfig(
                feat_extractor="BertaFeatureExtractor",
                indicator_columns=context_indicator_columns,
            ),
            tto_lib_loader_config=LibLoaderConfig(
                root_dirs=[annotated_speclib_path],
                frag_types=charged_frag_types,
            ),
        )
        self._tto_manager = TTOManager(tto_config)

    def run(self, save_path: str) -> None:
        """Extract context from the annotated spectral library and save to disk.

        Parameters
        ----------
        save_path : str
            Full output path for the context including the extension.
        """
        self._tto_manager.run(save_path)
