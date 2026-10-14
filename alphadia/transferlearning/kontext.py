"""Adapt peptdeep_kontext to the data of a search.

This is the alternative to the transfer learning module. Test-time optimization (TTO)
learns a context vector for the data while the pretrained model stays frozen. The
"hybrid" method additionally runs transfer learning on the downstream model, using
that context. All precursors share a single context.

peptdeep_kontext is imported lazily, so alphaDIA can run without it unless it is used.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from alphadia import utils
from alphadia.exceptions import GenericUserError

logger = logging.getLogger(__name__)

ADAPTATION_TTO = "tto"
ADAPTATION_HYBRID = "hybrid"

# peptdeep_kontext model classes; each model is saved in a directory of the same name
CONTEXT_MODEL = "ContextModel"
DOWNSTREAM_MODEL = "ContextDownstream"
FEATURE_EXTRACTOR = "BertaFeatureExtractor"

# written by peptdeep_kontext into the directory a training run is given
_CONTEXT_FILE_NAME = "context.json"
_TRANSFER_LEARNING_DIR = "transfer_learning"

_DEFAULT_MODEL_CACHE = utils.expand_path("~/.cache/peptdeep_kontext")


def get_context_path(adaptation_dir: str | Path) -> str:
    """Path of the context learned by TTO in `adaptation_dir`."""
    return os.path.join(adaptation_dir, _CONTEXT_FILE_NAME)


def get_transfer_learned_model_path(adaptation_dir: str | Path) -> str:
    """Path of the downstream model trained by the hybrid method in `adaptation_dir`."""
    return os.path.join(adaptation_dir, _TRANSFER_LEARNING_DIR, DOWNSTREAM_MODEL)


def resolve_model_dir(kontext_config: dict) -> str:
    """Return the directory of the pretrained models, downloading them if `model_dir` is not set."""
    if kontext_config["model_dir"]:
        return utils.expand_path(kontext_config["model_dir"])

    from peptdeep_kontext.utils.cfg import download_pretrained_models

    version = kontext_config["model_version"]
    logger.info(f"Using peptdeep_kontext weights {version} from {_DEFAULT_MODEL_CACHE}")
    return download_pretrained_models(target_dir=_DEFAULT_MODEL_CACHE, version=version)


def load_config_file(config_path: str | None) -> dict:
    """Load a peptdeep_kontext pipeline config from a .py or .yaml file; empty if no path is given."""
    if not config_path:
        return {}

    config_path = utils.expand_path(config_path)
    if config_path.endswith(".py"):
        from peptdeep_kontext.utils.cfg import load_python_config

        return load_python_config(config_path)

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_torch_device(use_gpu: bool) -> str:
    """Torch device name for peptdeep_kontext; alphaDIA's own helper says "gpu" where torch expects "cuda"."""
    device = utils.get_torch_device(use_gpu)
    return "cuda" if device == "gpu" else device


def build_config(
    config_class: type,
    alphadia_settings: dict,
    file_settings: dict,
    fixed_settings: dict,
) -> Any:
    """Build a peptdeep_kontext config object.

    Settings are applied in increasing order of precedence: the peptdeep_kontext defaults,
    the settings from the alphaDIA config, the settings from the peptdeep_kontext config file,
    and the settings alphaDIA always sets itself (paths, models, device, ...).

    Parameters
    ----------
    config_class : type
        peptdeep_kontext config class, e.g. `TrainingConfig` or `ModelManagerConfig`.
    alphadia_settings : dict
        Settings from the alphaDIA config.
    file_settings : dict
        The matching section of the peptdeep_kontext config file.
    fixed_settings : dict
        Settings alphaDIA always sets itself.

    Raises
    ------
    GenericUserError
        If a setting from the alphaDIA config or the config file does not exist in `config_class`.
    """
    from peptdeep_kontext.cli import find_unknown_keys

    config = config_class()

    # only user settings are checked: pretrained model configs may contain keys the current code ignores
    for settings in (alphadia_settings, file_settings):
        unknown = find_unknown_keys(config, settings)
        if unknown:
            raise GenericUserError(
                f"Unknown peptdeep_kontext settings for {config_class.__name__}: {', '.join(unknown)}"
            )

    # later settings take precedence; load_attrs merges nested configs and only sets the given keys
    for settings in (alphadia_settings, file_settings, fixed_settings):
        config.load_attrs(settings)
    return config


def run_adaptation(
    method: str,
    transfer_library_path: str,
    output_dir: str,
    kontext_config: dict,
    charged_frag_types: list[str],
    use_gpu: bool,
) -> None:
    """Adapt peptdeep_kontext to the transfer library and write the results to `output_dir`.

    Writes the context to `get_context_path(output_dir)` and, for the hybrid method,
    the transfer-learned model to `get_transfer_learned_model_path(output_dir)`.

    Parameters
    ----------
    method : str
        "tto" or "hybrid".
    transfer_library_path : str
        Path to the transfer library, i.e. the precursors and fragments identified in the search.
    output_dir : str
        Directory to write the results to.
    kontext_config : dict
        The `peptdeep_kontext` section of the alphaDIA config.
    charged_frag_types : list[str]
        Charged fragment types of the transfer library to train on.
    use_gpu : bool
        Whether to use the GPU if available.
    """
    from peptdeep_kontext.core.training import TrainingConfig, TrainingManager
    from peptdeep_kontext.utils.cfg import get_pretrained_model_dict_from_dir

    if method not in (ADAPTATION_TTO, ADAPTATION_HYBRID):
        raise ValueError(f"Unknown peptdeep_kontext adaptation method '{method}'")

    model_dir = resolve_model_dir(kontext_config)
    file_config = load_config_file(kontext_config["config_path"])

    data_settings = {
        "lib_loader_config": {
            "root_dirs": [transfer_library_path],
            "frag_types": charged_frag_types,
        },
        "dataset_config": {
            "feat_extractor": FEATURE_EXTRACTOR,
            "indicator_columns": None,
        },
        "device": get_torch_device(use_gpu),
    }

    logger.info("Learning the context with peptdeep_kontext TTO")
    # TTO learns only the context; the pretrained ContextModel stays frozen
    tto_fixed_settings = data_settings | {
        "model_class": CONTEXT_MODEL,
        "pretrained_model": get_pretrained_model_dict_from_dir(
            os.path.join(model_dir, CONTEXT_MODEL)
        ),
        "model_trainable": "none",
        "context_mode": "learned",
    }
    tto_config = build_config(
        TrainingConfig,
        alphadia_settings={"verbose": True}
        | _settings_with_prefix(kontext_config, "tto_"),
        file_settings=file_config.get("tto_config", {}),
        fixed_settings=tto_fixed_settings,
    )
    _save_config(tto_config, os.path.join(output_dir, "tto_config.yaml"))
    TrainingManager(tto_config).run(save_dir=output_dir)

    if method == ADAPTATION_HYBRID:
        logger.info("Transfer learning with peptdeep_kontext")
        transfer_learning_dir = os.path.join(output_dir, _TRANSFER_LEARNING_DIR)
        # trains the pretrained ContextDownstream further, using the context learned by TTO
        transfer_learning_fixed_settings = data_settings | {
            "model_class": DOWNSTREAM_MODEL,
            "pretrained_model": get_pretrained_model_dict_from_dir(
                os.path.join(model_dir, DOWNSTREAM_MODEL)
            ),
            "context_mode": "precomputed",
            "context_source_path": get_context_path(output_dir),
        }
        transfer_learning_config = build_config(
            TrainingConfig,
            alphadia_settings={"verbose": True}
            | _settings_with_prefix(kontext_config, "transfer_learning_"),
            file_settings=file_config.get("downstream_finetune_config", {}),
            fixed_settings=transfer_learning_fixed_settings,
        )
        _save_config(
            transfer_learning_config,
            os.path.join(output_dir, "transfer_learning_config.yaml"),
        )
        TrainingManager(transfer_learning_config).run(save_dir=transfer_learning_dir)


def _save_config(config: Any, path: str) -> None:
    """Save the resolved config, as the frozen alphaDIA config does not include the config file's content."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)


def _settings_with_prefix(kontext_config: dict, prefix: str) -> dict:
    """Settings of one training stage, e.g. {"epochs": 10} for `tto_epochs: 10` and prefix "tto_"."""
    return {
        key.removeprefix(prefix): value
        for key, value in kontext_config.items()
        if key.startswith(prefix)
    }
