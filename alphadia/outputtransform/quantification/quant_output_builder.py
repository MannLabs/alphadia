import logging
import os

import pandas as pd

from alphadia.constants.keys import (
    INTERNAL_TO_OUTPUT_MAPPING,
    NormalizationMethods,
    QuantificationLevelKey,
    QuantificationLevelName,
)
from alphadia.outputtransform.quantification.fragment_accumulator import (
    FragmentQuantLoader,
)
from alphadia.outputtransform.quantification.quant_builder import (
    LFQOutputConfig,
    QuantBuilder,
)
from alphadia.outputtransform.utils import merge_quant_levels_to_psm

logger = logging.getLogger()


class QuantOutputBuilder:
    """Build quantification outputs at multiple levels (precursor, peptide, protein).

    This class orchestrates the accumulation of fragment data, filtering by quality,
    and label-free quantification using directLFQ at different aggregation levels.

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
    DEFAULT_COLUMNS = ["intensity", "correlation"]

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
        psm_no_decoys = psm_df[psm_df["decoy"] == 0]

        if (
            config["search_output"]["normalization_method"]
            == NormalizationMethods.QUANTSELECT
        ):
            columns = self.QUANTSELECT_COLUMNS
        else:
            columns = self.DEFAULT_COLUMNS

        self.fragment_loader = FragmentQuantLoader(psm_no_decoys, columns=columns)
        self.quant_builder = QuantBuilder(psm_no_decoys, columns=columns)

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
            - Dictionary with quantification results for each level
            - Updated PSM dataframe with merged quantification data
        """
        logger.info("Performing label free quantification")

        feature_dfs_dict = self.fragment_loader.accumulate_from_folders(folder_list)

        if feature_dfs_dict is None or not feature_dfs_dict:
            logger.warning("No fragment data found, skipping quantification")
            return {}, self.psm_df

        quantlevel_configs = self._create_quant_level_configs()
        lfq_results = {}

        for quantlevel_config in quantlevel_configs:
            if not quantlevel_config.should_process:
                continue

            logger.info(
                f"Performing label free quantification on the {quantlevel_config.level_name} level"
            )

            lfq_df = self._process_quant_level(
                lfq_config=quantlevel_config, feature_dfs_dict=feature_dfs_dict
            )

            if lfq_df is not None and not lfq_df.empty:
                lfq_df = self._annotate_quant_df(lfq_df, self.psm_df, quantlevel_config)
                lfq_results[quantlevel_config.level_name] = lfq_df
            else:
                logger.warning(
                    f"No fragments found for {quantlevel_config.level_name}, skipping label-free quantification"
                )

        psm_df_with_quant = merge_quant_levels_to_psm(
            self.psm_df, lfq_results, quantlevel_configs
        )

        return lfq_results, psm_df_with_quant

    def _create_quant_level_configs(self) -> list[LFQOutputConfig]:
        """Create quantification level configurations based on settings.

        Uses output keys for intensity columns to ensure consistent
        output column naming in user-facing files.

        Returns
        -------
        list[LFQOutputConfig]
            List of quantification level configurations
        """
        return [
            LFQOutputConfig(
                quant_level=QuantificationLevelKey.PRECURSOR,
                level_name=QuantificationLevelName.PRECURSOR,
                intensity_column="precursor_lfq_intensity",
                aggregation_components=[
                    QuantificationLevelName.PROTEIN,
                    "sequence",
                    "mods",
                    "mod_sites",
                    "charge",
                ],
                should_process=self.config["search_output"]["precursor_level_lfq"],
                save_fragments=self.config["search_output"][
                    "save_fragment_quant_matrix"
                ],
                normalization_method=self.config["search_output"][
                    "normalization_method"
                ],
            ),
            LFQOutputConfig(
                quant_level=QuantificationLevelKey.PEPTIDE,
                level_name=QuantificationLevelName.PEPTIDE,
                intensity_column="peptide_lfq_intensity",
                aggregation_components=[
                    QuantificationLevelName.PROTEIN,
                    "sequence",
                    "mods",
                    "mod_sites",
                ],
                should_process=self.config["search_output"]["peptide_level_lfq"],
                save_fragments=self.config["search_output"][
                    "save_fragment_quant_matrix"
                ],
                normalization_method=self.config["search_output"][
                    "normalization_method"
                ],
            ),
            LFQOutputConfig(
                quant_level=QuantificationLevelKey.PROTEIN,
                level_name=QuantificationLevelName.PROTEIN,
                intensity_column="pg_lfq_intensity",
                aggregation_components=[
                    QuantificationLevelName.PROTEIN,
                ],
                should_process=True,
                normalization_method=self.config["search_output"][
                    "normalization_method"
                ],
            ),
        ]

    def _annotate_quant_df(
        self,
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

    def _process_quant_level(
        self,
        lfq_config: LFQOutputConfig,
        feature_dfs_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame | None:
        """Process quantification for a single level.

        Parameters
        ----------
        lfq_config : LFQOutputConfig
            Configuration for this quantification level
        feature_dfs_dict : dict[str, pd.DataFrame]
            Dictionary with feature name as key and a df as value, where df is a feature dataframe with the columns precursor_idx, ion, raw_name1, raw_name2, ...

        Returns
        -------
        pd.DataFrame | None
            Quantification results, or None if no data available
        """
        if lfq_config.normalization_method == NormalizationMethods.DIRECTLFQ:
            filtered_intensity_df, _ = self.quant_builder.filter_frag_df(
                feature_dfs_dict["intensity"],
                feature_dfs_dict["correlation"],
                top_n=self.config["search_output"]["min_k_fragments"],
                min_correlation=self.config["search_output"]["min_correlation"],
                group_column=lfq_config.quant_level,
            )

            if len(filtered_intensity_df) == 0:
                logger.warning(
                    f"No fragments found for {lfq_config.level_name}, skipping label-free quantification"
                )
                return None

            lfq_df = self.quant_builder.direct_lfq(
                intensity_df=filtered_intensity_df,
                lfq_config=lfq_config,
                config=self.config,
            )

        else:
            lfq_df = self.quant_builder.quantselect_lfq(
                feature_dfs_dict=feature_dfs_dict,
                lfq_config=lfq_config,
            )
        return lfq_df

    def _apply_output_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert internal column names to output names for output.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with internal column names

        Returns
        -------
        pd.DataFrame
            Dataframe with output column names applied
        """
        return df.rename(columns=INTERNAL_TO_OUTPUT_MAPPING)

    def save_results(
        self,
        lfq_results: dict[str, pd.DataFrame],
        output_folder: str,
        file_format: str = "parquet",
    ) -> None:
        """Save quantification results to disk with output column names.

        Parameters
        ----------
        lfq_results : dict[str, pd.DataFrame]
            Dictionary with quantification results for each level
        output_folder : str
            Output folder path
        file_format : str, default='parquet'
            File format for output files
        """
        from alphadia.outputtransform.utils import write_df

        quantlevel_configs = self._create_quant_level_configs()

        for config in quantlevel_configs:
            if not config.should_process:
                continue

            lfq_df = lfq_results.get(config.level_name)
            if lfq_df is None or lfq_df.empty:
                continue

            logger.info(f"Writing {config.level_name} output to disk")

            lfq_df_output = self._apply_output_names(lfq_df)

            write_df(
                lfq_df_output,
                os.path.join(output_folder, f"{config.level_name}.matrix"),
                file_format=file_format,
            )

    def save_fragment_matrices(
        self,
        feature_dfs_dict: dict[str, pd.DataFrame],
        output_folder: str,
        file_format: str = "parquet",
    ) -> None:
        """Save fragment-level quantification matrices to disk with output column names.

        Parameters
        ----------
        feature_dfs_dict : dict[str, pd.DataFrame]
            Dictionary containing intensity and correlation dataframes
        output_folder : str
            Output folder path
        file_format : str, default='parquet'
            File format for output files
        """
        from alphadia.outputtransform.utils import write_df

        quantlevel_configs = self._create_quant_level_configs()

        for config in quantlevel_configs:
            if not config.save_fragments:
                continue

            group_intensity_df, _ = self.quant_builder.filter_frag_df(
                feature_dfs_dict["intensity"],
                feature_dfs_dict["correlation"],
                top_n=self.config["search_output"]["min_k_fragments"],
                min_correlation=self.config["search_output"]["min_correlation"],
                group_column=config.quant_level,
            )

            if len(group_intensity_df) == 0:
                continue

            logger.info(
                f"Writing fragment quantity matrix to disk, filtered on {config.level_name}"
            )

            group_intensity_df_output = self._apply_output_names(group_intensity_df)

            write_df(
                group_intensity_df_output,
                os.path.join(
                    output_folder, f"fragment_{config.level_name}filtered.matrix"
                ),
                file_format=file_format,
            )
