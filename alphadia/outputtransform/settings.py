"""Settings of the search plan output, parsed once from the config.

The output modules only read these settings, so the config keys they depend on are
listed in one place and a missing key fails when the output is set up, not midway
through building it.
"""

from dataclasses import dataclass

from alphabase.peptide import fragment

from alphadia.outputtransform.utils import get_channels_from_config
from alphadia.workflow.config import Config


@dataclass(frozen=True)
class FdrSettings:
    """Protein inference and protein FDR settings."""

    fdr: float
    inference_strategy: str
    group_level: str
    keep_decoys: bool
    keep_decoys_in_mbr_library: bool
    decoy_type: str

    @classmethod
    def from_config(cls, config: Config) -> "FdrSettings":
        fdr_config = config["fdr"]
        return cls(
            fdr=fdr_config["fdr"],
            inference_strategy=fdr_config["inference_strategy"],
            group_level=fdr_config["group_level"],
            keep_decoys=fdr_config["keep_decoys"],
            keep_decoys_in_mbr_library=fdr_config["keep_decoys_in_mbr_library"],
            decoy_type=fdr_config["decoy_type"],
        )


@dataclass(frozen=True)
class LfqSettings:
    """Label-free quantification settings."""

    normalization_method: str
    intensity_drift_correction: bool
    precursor_level_lfq: bool
    peptide_level_lfq: bool
    min_k_fragments: int
    min_correlation: float
    normalize_directlfq: bool
    min_nonnan: int
    num_samples_quadratic: int
    thread_count: int

    @classmethod
    def from_config(cls, config: Config) -> "LfqSettings":
        search_output = config["search_output"]
        return cls(
            normalization_method=search_output["normalization_method"].lower(),
            intensity_drift_correction=search_output["intensity_drift_correction"],
            precursor_level_lfq=search_output["precursor_level_lfq"],
            peptide_level_lfq=search_output["peptide_level_lfq"],
            min_k_fragments=search_output["min_k_fragments"],
            min_correlation=search_output["min_correlation"],
            normalize_directlfq=search_output["normalize_directlfq"],
            min_nonnan=search_output["min_nonnan"],
            num_samples_quadratic=search_output["num_samples_quadratic"],
            thread_count=config["general"]["thread_count"],
        )


@dataclass(frozen=True)
class TransferLibrarySettings:
    """Settings for building the transfer learning library."""

    enabled: bool
    top_k_samples: int
    norm_delta_max: bool
    precursor_correlation_cutoff: float
    fragment_correlation_ratio: float
    charged_frag_types: list[str]

    @classmethod
    def from_config(cls, config: Config) -> "TransferLibrarySettings":
        transfer_library = config["transfer_library"]
        return cls(
            enabled=transfer_library["enabled"],
            top_k_samples=transfer_library["top_k_samples"],
            norm_delta_max=transfer_library["norm_delta_max"],
            precursor_correlation_cutoff=transfer_library[
                "precursor_correlation_cutoff"
            ],
            fragment_correlation_ratio=transfer_library["fragment_correlation_ratio"],
            charged_frag_types=fragment.get_charged_frag_types(
                transfer_library["fragment_types"],
                transfer_library["max_charge"],
            ),
        )


@dataclass(frozen=True)
class TransferLearningSettings:
    """Settings for finetuning the PeptDeep models on the transfer library."""

    enabled: bool
    use_gpu: bool
    lr_patience: int
    test_interval: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    epochs: int
    warmup_epochs: int
    batch_size: int
    max_lr: float
    nce: float
    instrument: str

    @classmethod
    def from_config(cls, config: Config) -> "TransferLearningSettings":
        transfer_learning = config["transfer_learning"]
        return cls(
            enabled=transfer_learning["enabled"],
            use_gpu=config["general"]["use_gpu"],
            lr_patience=transfer_learning["lr_patience"],
            test_interval=transfer_learning["test_interval"],
            train_fraction=transfer_learning["train_fraction"],
            validation_fraction=transfer_learning["validation_fraction"],
            test_fraction=transfer_learning["test_fraction"],
            epochs=transfer_learning["epochs"],
            warmup_epochs=transfer_learning["warmup_epochs"],
            batch_size=transfer_learning["batch_size"],
            max_lr=transfer_learning["max_lr"],
            nce=transfer_learning["nce"],
            instrument=transfer_learning["instrument"],
        )


@dataclass(frozen=True)
class OutputSettings:
    """All settings `SearchPlanOutput` reads from the config."""

    file_format: str
    save_figures: bool
    save_mbr_library: bool
    channels: list[int]
    fdr: FdrSettings
    lfq: LfqSettings
    transfer_library: TransferLibrarySettings
    transfer_learning: TransferLearningSettings

    @classmethod
    def from_config(cls, config: Config) -> "OutputSettings":
        return cls(
            file_format=config["search_output"]["file_format"],
            save_figures=config["general"]["save_figures"],
            save_mbr_library=config["general"]["save_mbr_library"],
            channels=get_channels_from_config(config),
            fdr=FdrSettings.from_config(config),
            lfq=LfqSettings.from_config(config),
            transfer_library=TransferLibrarySettings.from_config(config),
            transfer_learning=TransferLearningSettings.from_config(config),
        )
