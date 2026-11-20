import logging

import numpy as np
import pandas as pd

from alphadia.constants.keys import (
    NormalizationMethods,
    QuantificationLevelKey,
    QuantificationLevelName,
)
from alphadia.outputtransform.quantification.fragment_accumulator import (
    FragmentQuantLoader,
)
from alphadia.outputtransform.quantification.intensity_drift_correction import (
    correct_intensity_drift,
    correct_precursor_intensity_drift,
)
from alphadia.outputtransform.quantification.quant_builder import (
    DEFAULT_FEATURE_COLUMNS,
    INTENSITY_LIBRARY_COLUMN,
    LFQOutputConfig,
    direct_lfq,
    filter_frag_df,
    quantselect_lfq,
    varmodel_precursor_table,
)

logger = logging.getLogger()

# a precursor is already the robust estimate over its fragments, so a single one suffices
MIN_PRECURSORS_PER_RUN = 1

QUANT_LEVEL_CONFIGS: dict[str, LFQOutputConfig] = {
    QuantificationLevelName.PRECURSOR: LFQOutputConfig(
        quant_level=QuantificationLevelKey.PRECURSOR,
        level_name=QuantificationLevelName.PRECURSOR,
        intensity_column="precursor_lfq_intensity",
        aggregation_components=[
            QuantificationLevelKey.PROTEIN,
            "sequence",
            "mods",
            "mod_sites",
            "charge",
        ],
    ),
    QuantificationLevelName.PEPTIDE: LFQOutputConfig(
        quant_level=QuantificationLevelKey.PEPTIDE,
        level_name=QuantificationLevelName.PEPTIDE,
        intensity_column="peptide_lfq_intensity",
        aggregation_components=[
            QuantificationLevelKey.PROTEIN,
            "sequence",
            "mods",
            "mod_sites",
        ],
    ),
    QuantificationLevelName.PROTEIN: LFQOutputConfig(
        quant_level=QuantificationLevelKey.PROTEIN,
        level_name=QuantificationLevelName.PROTEIN,
        intensity_column="pg_lfq_intensity",
        aggregation_components=[QuantificationLevelKey.PROTEIN],
    ),
}

# the protein level is always quantified, the others can be switched off
LEVEL_ENABLED_CONFIG_KEYS = {
    QuantificationLevelName.PRECURSOR: "precursor_level_lfq",
    QuantificationLevelName.PEPTIDE: "peptide_level_lfq",
}


class QuantOutputBuilder:
    """Build quantification outputs at multiple levels (precursor, peptide, protein).

    This class orchestrates the accumulation of fragment data, filtering by quality,
    and label-free quantification at different aggregation levels.

    Parameters
    ----------
    psm_df : pd.DataFrame
        PSM dataframe containing precursor identifications
    config : dict
        Configuration dictionary with quantification settings
    """

    QUANTSELECT_COLUMNS = [
        "intensity",
        "mass_error",
        "correlation",
        "charge",
        "mz_observed",
        "type",
        "number",
        "cardinality",
        "position",
    ]

    def __init__(self, psm_df: pd.DataFrame, config: dict):
        """Initialize the QuantOutputBuilder.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM dataframe with precursor identifications
        config : dict
            Configuration dictionary containing search_output and general settings
        """
        self.psm_df = psm_df
        self.config = config
        self._psm_no_decoys_df = psm_df[psm_df["decoy"] == 0]
        self._normalization_method = config["search_output"][
            "normalization_method"
        ].lower()

        columns = (
            self.QUANTSELECT_COLUMNS
            if self._normalization_method == NormalizationMethods.QUANTSELECT
            else DEFAULT_FEATURE_COLUMNS
        )
        constant_feature_columns = (
            [INTENSITY_LIBRARY_COLUMN]
            if self._normalization_method == NormalizationMethods.VARMODEL_DIRECTLFQ
            else []
        )

        self.fragment_loader = FragmentQuantLoader(
            self._psm_no_decoys_df,
            columns=columns,
            constant_feature_columns=constant_feature_columns,
        )

    def build(
        self, folder_list: list[str]
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Build quantification outputs at multiple levels.

        Parameters
        ----------
        folder_list : list[str]
            List of folders containing fragment data files

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            Tuple containing:
            - Annotated quantification table by level name, for every requested level
            - PSM dataframe with the quantities of every level merged in
        """
        logger.info("Performing label free quantification")

        feature_dfs_dict = self.fragment_loader.accumulate_from_folders(folder_list)

        requested_configs = self._requested_quant_level_configs()

        # TODO cover the QuantSelect branch with a build-level test: every requested
        #  level is routed to quantselect_lfq and the directLFQ fragment filter is skipped
        if self._normalization_method == NormalizationMethods.QUANTSELECT:
            level_dfs = self._quantify_with_quantselect(
                requested_configs, feature_dfs_dict
            )
        elif self._normalization_method == NormalizationMethods.VARMODEL_DIRECTLFQ:
            level_dfs = self._quantify_with_varmodel_precursors(
                requested_configs, feature_dfs_dict
            )
        else:
            level_dfs = self._quantify_with_directlfq(
                requested_configs, feature_dfs_dict
            )

        # decoys carry the precursor and peptide hashes of their targets, so they could annotate a target's row
        lfq_results = {
            level_name: _annotate_quant_df(
                lfq_df, self._psm_no_decoys_df, QUANT_LEVEL_CONFIGS[level_name]
            )
            for level_name, lfq_df in level_dfs.items()
        }

        return lfq_results, merge_quant_levels_to_psm(self.psm_df, level_dfs)

    def _is_drift_corrected(self) -> bool:
        return self.config["search_output"]["intensity_drift_correction"]

    def _drift_corrected_fragments(
        self, feature_dfs_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Return `feature_dfs_dict` with the intensity drift removed from the fragment intensities if enabled."""
        if not self._is_drift_corrected():
            return feature_dfs_dict
        return {
            **feature_dfs_dict,
            "intensity": correct_intensity_drift(
                feature_dfs_dict["intensity"],
                feature_dfs_dict["correlation"],
                self._psm_no_decoys_df,
            ),
        }

    def _requested_quant_level_configs(self) -> list[LFQOutputConfig]:
        """Quantification levels switched on in the config."""
        return [
            level_config
            for level_name, level_config in QUANT_LEVEL_CONFIGS.items()
            if level_name not in LEVEL_ENABLED_CONFIG_KEYS
            or self.config["search_output"][LEVEL_ENABLED_CONFIG_KEYS[level_name]]
        ]

    def _quantify_with_quantselect(
        self,
        quantlevel_configs: list[LFQOutputConfig],
        feature_dfs_dict: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Remove the fragment intensity drift if enabled and quantify every requested level with QuantSelect.

        Parameters
        ----------
        quantlevel_configs : list[LFQOutputConfig]
            Requested quantification levels
        feature_dfs_dict : dict[str, pd.DataFrame]
            Dictionary with feature name as key and a df as value, where df is a feature dataframe with the columns precursor_idx, ion, raw_name1, raw_name2, ...

        Returns
        -------
        dict[str, pd.DataFrame]
            Quantification results by level name
        """
        feature_dfs_dict = self._drift_corrected_fragments(feature_dfs_dict)
        level_dfs = {}
        for quantlevel_config in quantlevel_configs:
            logger.info(
                f"Performing label free quantification on the {quantlevel_config.level_name} level"
            )
            level_dfs[quantlevel_config.level_name] = quantselect_lfq(
                psm_df=self._psm_no_decoys_df,
                feature_dfs_dict=feature_dfs_dict,
                lfq_config=quantlevel_config,
            )
        return level_dfs

    def _quantify_with_directlfq(
        self,
        quantlevel_configs: list[LFQOutputConfig],
        feature_dfs_dict: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Remove the fragment intensity drift if enabled, filter fragments per level and estimate the level intensities with directLFQ.

        Parameters
        ----------
        quantlevel_configs : list[LFQOutputConfig]
            Requested quantification levels
        feature_dfs_dict : dict[str, pd.DataFrame]
            Dictionary with feature name as key and a df as value, where df is a feature dataframe with the columns precursor_idx, ion, raw_name1, raw_name2, ...

        Returns
        -------
        dict[str, pd.DataFrame]
            Quantification results by level name
        """
        feature_dfs_dict = self._drift_corrected_fragments(feature_dfs_dict)
        level_dfs = {}
        for quantlevel_config in quantlevel_configs:
            logger.info(
                f"Performing label free quantification on the {quantlevel_config.level_name} level"
            )
            filtered_intensity_df = self._filter_fragments(
                feature_dfs_dict, quantlevel_config.quant_level
            )
            level_dfs[quantlevel_config.level_name] = direct_lfq(
                intensity_df=filtered_intensity_df,
                lfq_config=quantlevel_config,
                config=self.config,
            )
        return level_dfs

    def _quantify_with_varmodel_precursors(
        self,
        quantlevel_configs: list[LFQOutputConfig],
        feature_dfs_dict: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Estimate the precursor intensities with the variance-model fit, remove their intensity drift if enabled and roll them up with directLFQ.

        The drift is corrected on the precursor table, along the retention time and the neutral
        precursor mass, not on the fragments. All levels are rolled up from the same precursor
        table so that they share the precursor intensities; directLFQ's sample normalization does
        not depend on the grouping, so every level also gets the same shifts.

        Parameters
        ----------
        quantlevel_configs : list[LFQOutputConfig]
            Requested quantification levels
        feature_dfs_dict : dict[str, pd.DataFrame]
            Dictionary with feature name as key and a df as value, where df is a feature dataframe with the columns precursor_idx, ion, raw_name1, raw_name2, ...

        Returns
        -------
        dict[str, pd.DataFrame]
            Quantification results by level name
        """
        precursor_df = varmodel_precursor_table(
            self._filter_fragments(feature_dfs_dict, QuantificationLevelKey.PRECURSOR)
        )
        if self._is_drift_corrected():
            precursor_df = correct_precursor_intensity_drift(
                precursor_df, self._psm_no_decoys_df
            )

        level_dfs = {}
        for quantlevel_config in quantlevel_configs:
            logger.info(
                f"Performing label free quantification on the {quantlevel_config.level_name} level"
            )
            level_dfs[quantlevel_config.level_name] = direct_lfq(
                intensity_df=precursor_df,
                lfq_config=quantlevel_config,
                config=self.config,
                min_nonan=MIN_PRECURSORS_PER_RUN,
            )
        return level_dfs

    def _filter_fragments(
        self, feature_dfs_dict: dict[str, pd.DataFrame], group_column: str
    ) -> pd.DataFrame:
        """Fragment intensities that pass the correlation filter within each group."""
        return filter_frag_df(
            feature_dfs_dict["intensity"],
            feature_dfs_dict["correlation"],
            min_correlation=self.config["search_output"]["min_correlation"],
            top_n=self.config["search_output"]["min_k_fragments"],
            group_column=group_column,
        )


def _annotate_quant_df(
    lfq_df: pd.DataFrame,
    psm_df: pd.DataFrame,
    config: LFQOutputConfig,
) -> pd.DataFrame:
    """Annotate quantification results with metadata from PSM dataframe.

    Parameters
    ----------
    lfq_df : pd.DataFrame
        Quantification results dataframe
    psm_df : pd.DataFrame
        PSM dataframe containing metadata
    config : LFQOutputConfig
        Configuration specifying grouping and aggregation columns

    Returns
    -------
    pd.DataFrame
        Annotated quantification dataframe
    """
    if config.level_name == QuantificationLevelName.PROTEIN:
        return lfq_df

    annotate_df = psm_df.groupby(config.quant_level, as_index=False).agg(
        {c: "first" for c in config.aggregation_components}
    )
    return lfq_df.merge(annotate_df, on=config.quant_level, how="left")


def merge_quant_levels_to_psm(
    psm_df: pd.DataFrame,
    level_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge quantification results from all levels back to the precursor table.

    The quantities are estimated from the targets only. Decoys carry the hashes of their
    targets, so they are excluded from the merge and keep no quantities.

    Parameters
    ----------
    psm_df : pd.DataFrame
        Precursor table to merge quantification data into
    level_dfs : dict[str, pd.DataFrame]
        Quantification table by level name, with the quant level column and one column per run

    Returns
    -------
    pd.DataFrame
        Updated precursor table with merged quantification data
    """
    for level_name, lfq_df in level_dfs.items():
        level_config = QUANT_LEVEL_CONFIGS[level_name]

        melted_df = lfq_df.melt(
            id_vars=level_config.quant_level,
            var_name="run",
            value_name=level_config.intensity_column,
        )
        # the dtype of the decoy column varies, a zero of the same dtype matches the targets only
        melted_df["decoy"] = np.zeros(len(melted_df), dtype=psm_df["decoy"].dtype)
        psm_df = psm_df.merge(
            melted_df, on=[level_config.quant_level, "run", "decoy"], how="left"
        )

    return psm_df
