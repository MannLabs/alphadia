import logging

import directlfq.config as lfqconfig
import directlfq.normalization as lfqnorm
import directlfq.protein_intensity_estimation as lfqprot_estimation
import directlfq.utils as lfqutils
import numba as nb
import numpy as np
import pandas as pd
from quantselect.config import QuantSelectConfig
from quantselect.dataloader import DataLoader
from quantselect.loader import Loader
from quantselect.ms1_features import FeatureConfig
from quantselect.preprocessing import PreprocessingPipeline
from quantselect.utils import set_global_determinism
from quantselect.var_model import Model

from alphadia.constants.keys import NormalizationMethods
from alphadia.outputtransform.quantification.quant_output_builder import LFQOutputConfig
from alphadia.utils import USE_NUMBA_CACHING

logger = logging.getLogger()


@nb.njit(cache=USE_NUMBA_CACHING)
def _ion_hash(precursor_idx, number, type, charge, loss_type):
    """Create a 64-bit hash from fragment ion characteristics.

    Parameters
    ----------
    precursor_idx : array-like
        Precursor indices (lower 32 bits)
    number : array-like
        Fragment number (next 8 bits)
    type : array-like
        Fragment type (next 8 bits)
    charge : array-like
        Fragment charge (next 8 bits)
    loss_type : array-like
        Loss type (last 8 bits)

    Returns
    -------
    int64
        64-bit hash value
    """
    return (
        precursor_idx
        + (number << 32)
        + (type << 40)
        + (charge << 48)
        + (loss_type << 56)
    )


def prepare_df(
    df: pd.DataFrame, psm_df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Prepare fragment dataframe by filtering and adding ion hash.

    Parameters
    ----------
    df : pd.DataFrame
        Fragment dataframe
    psm_df : pd.DataFrame
        PSM dataframe with precursor_idx column
    columns : list[str]
        Columns to keep from fragment data

    Returns
    -------
    pd.DataFrame
        Filtered fragment dataframe with ion hash
    """
    df = df[df["precursor_idx"].isin(psm_df["precursor_idx"])].copy()
    df["ion"] = _ion_hash(
        df["precursor_idx"].values,
        df["number"].values,
        df["type"].values,
        df["charge"].values,
        df["loss_type"].values,
    )
    return df[["precursor_idx", "ion"] + columns]


class QuantBuilder:
    """Build quantification results through filtering and label-free quantification.

    This class focuses on fragment quality filtering and directLFQ-based
    protein quantification. Fragment data accumulation is handled by
    FragmentQuantLoader.

    Parameters
    ----------
    psm_df : pd.DataFrame
        PSM dataframe with precursor information
    columns : list[str] | None, default=None
        Columns to use for quantification. Defaults to ["intensity", "correlation"]
    """

    def __init__(self, psm_df: pd.DataFrame, columns: list[str] | None = None):
        self.psm_df = psm_df
        self.columns = ["intensity", "correlation"] if columns is None else columns

    def filter_frag_df(
        self,
        intensity_df: pd.DataFrame,
        quality_df: pd.DataFrame,
        min_correlation: float = 0.5,
        top_n: int = 3,
        group_column: str = "pg",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter fragment data by quality metrics.

        Keeps fragments that meet either of these criteria:
        - Among top N fragments per group (by mean quality across runs)
        - Quality score above min_correlation threshold

        Parameters
        ----------
        intensity_df : pd.DataFrame
            Fragment intensity data with columns: precursor_idx, ion, run1, run2, ..., pg, mod_seq_hash, mod_seq_charge_hash
        quality_df : pd.DataFrame
            Fragment quality/correlation data with same structure as intensity_df
        min_correlation : float, default=0.5
            Minimum quality score to keep fragment (if not in top N)
        top_n : int, default=3
            Number of top fragments to keep per group
        group_column : str, default='pg'
            Column to group fragments by (pg, mod_seq_hash, mod_seq_charge_hash)

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Filtered intensity and quality dataframes
        """
        logger.info("Filtering fragments by quality")

        # Extract sample/run columns (e.g., run_0, run_1, run_2)
        # These are all columns except metadata columns
        run_columns = [
            c
            for c in intensity_df.columns
            if c
            not in ["precursor_idx", "ion", "pg", "mod_seq_hash", "mod_seq_charge_hash"]
        ]

        quality_df["total"] = np.mean(quality_df[run_columns].values, axis=1)
        quality_df["rank"] = quality_df.groupby(group_column)["total"].rank(
            ascending=False, method="first"
        )
        mask = (quality_df["rank"].values <= top_n) | (
            quality_df["total"].values > min_correlation
        )
        return intensity_df[mask], quality_df[mask]

    def lfq(
        self,
        feature_dfs_dict: dict[str, pd.DataFrame],
        lfq_config: LFQOutputConfig,
        search_config: dict,
    ) -> pd.DataFrame:
        """Perform label-free quantification using directLFQ.

        Parameters
        ----------
        feature_dfs_dict: dict[str, pd.DataFrame]
            Dictionary with feature name as key and a df as value, where df is a feature dataframe with the columns precursor_idx, ion, raw_name1, raw_name2, ...
        lfq_config: LFQOutputConfig
            Configuration for this quantification level
        search_config: dict
            Global configuration dictionary
        Returns
        -------
        pd.DataFrame
            Protein/peptide quantification results with columns: group_column, run1, run2, ...
        """
        logger.info(
            f"Performing label-free quantification with {lfq_config.normalization_method} normalization"
        )

        # Apply normalization based on the selected method
        if lfq_config.normalization_method == NormalizationMethods.QUANT_SELECT:
            logger.info("Applying QuantSelect normalization")

            # Use provided config or default
            quantselect_config = QuantSelectConfig().CONFIG

            # Set random seed
            seed = quantselect_config.get("seed", 42)
            set_global_determinism(seed=seed)

            # Prepare MS1 features from PSM data
            precursor_df = Loader()._pivot_table_by_feature(
                FeatureConfig.DEFAULT_FEATURES, self.psm_df[self.psm_df["decoy"] == 0]
            )
            keys = list(feature_dfs_dict.keys())
            for k in keys:
                if "ms2" not in k:
                    feature_dfs_dict[f"ms2_{k}"] = feature_dfs_dict.pop(k)
            features = {
                "ms1": precursor_df,
                "ms2": feature_dfs_dict,
            }

            # Initialize preprocessing pipeline
            pipeline = PreprocessingPipeline(standardize=True)

            # Process data at specified level
            feature_layer, intensity_layer = pipeline.process(
                data=features, level=lfq_config.quant_level
            )

            # Create dataloader object
            dataloader = DataLoader(
                feature_layer=feature_layer, intensity_layer=intensity_layer
            )

            # Initialize model with configuration
            model, optimizer, criterion = Model.initialize_for_training(
                dataloader=dataloader,
                criterion_params=quantselect_config["criterion_params"],
                model_params=quantselect_config["model_params"],
                optimizer_params=quantselect_config["optmizer_params"],
            )

            # Train model with fit parameters from config
            fit_params = quantselect_config["fit_params"]
            model.fit(
                criterion=criterion,
                optimizer=optimizer,
                dataloader=dataloader,
                fit_params=fit_params,
            )

            # Generate predictions with configurable parameters
            normalized_data = model.predict(
                dataloader=dataloader,
                cutoff=0.9,
                min_num_fragments=12,
                no_const=3000,
            )

            # Convert from log2 space back to linear

            return (2**normalized_data).reset_index(names=lfq_config.quant_level)

        group_intensity_df, _ = self.filter_frag_df(
            feature_dfs_dict["intensity"],
            feature_dfs_dict["correlation"],
            top_n=search_config["search_output"]["min_k_fragments"],
            min_correlation=search_config["search_output"]["min_correlation"],
            group_column=lfq_config.quant_level,
        )

        if len(group_intensity_df) == 0:
            logger.warning(
                f"No fragments found for {lfq_config.level_name}, skipping label-free quantification"
            )
            return None

        # drop all other columns as they will be interpreted as samples
        columns_to_drop = list(
            {"precursor_idx", "pg", "mod_seq_hash", "mod_seq_charge_hash"}
            - {lfq_config.quant_level}
        )
        intensity_df = group_intensity_df.drop(columns=columns_to_drop)

        lfqconfig.set_global_protein_and_ion_id(
            protein_id=lfq_config.quant_level, quant_id="ion"
        )
        lfqconfig.set_compile_normalized_ion_table(compile_normalized_ion_table=False)
        lfqconfig.check_wether_to_copy_numpy_arrays_derived_from_pandas()
        lfqconfig.set_log_processed_proteins(log_processed_proteins=True)

        intensity_df.sort_values(
            by=lfq_config.quant_level, inplace=True, ignore_index=True
        )

        lfq_df = lfqutils.index_and_log_transform_input_df(intensity_df)
        lfq_df = lfqutils.remove_allnan_rows_input_df(lfq_df)

        if lfq_config.normalization_method == NormalizationMethods.DIRECT_LFQ:
            logger.info("Applying directLFQ normalization")
            lfq_df = lfqnorm.NormalizationManagerSamplesOnSelectedProteins(
                lfq_df,
                num_samples_quadratic=search_config["search_output"][
                    "num_samples_quadratic"
                ],
                selected_proteins_file=None,
            ).complete_dataframe

        protein_df, _ = lfqprot_estimation.estimate_protein_intensities(
            lfq_df,
            min_nonan=search_config["search_output"]["min_nonnan"],
            num_samples_quadratic=search_config["search_output"][
                "num_samples_quadratic"
            ],
            num_cores=search_config["general"]["thread_count"],
        )
        return protein_df
